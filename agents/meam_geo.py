import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, LogParam, MLP, TanhNormal, Value


class MEAMGeoAgent(flax.struct.PyTreeNode):
    """QAM with training-time Gaussian geometric expansion.

    The QAM critic, adjoint-matching objective, policy sampling, and target
    updates are kept unchanged. The only added mechanism is an auxiliary
    Gaussian actor that:

      1. is conditioned on (state, dataset action);
      2. predicts an absolute action, not a residual edit;
      3. is trained to maximize the frozen critic with SAC-style entropy;
      4. contributes its deterministic mode to a random fraction of the
         behavior-flow training targets; and
      5. is never used at inference time.

    Thus the slow flow learns an expanded prior, and ordinary QAM extracts the
    final policy relative to that expanded prior.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
        else:
            batch_actions = batch["actions"][..., 0, :]

        next_actions = self.sample_actions(
            batch["next_observations"][..., -1, :], rng=rng
        )
        next_actions = jnp.clip(next_actions, -1.0, 1.0)

        next_qs = self.network.select("target_critic")(
            batch["next_observations"][..., -1, :], next_actions
        )
        next_q = (
            next_qs.mean(axis=0)
            - self.config["rho"] * next_qs.std(axis=0)
        )

        target_q = (
            batch["rewards"][..., -1]
            + (self.config["discount"] ** self.config["horizon_length"])
            * batch["masks"][..., -1]
            * next_q
        )

        q = self.network.select("critic")(
            batch["observations"], batch_actions, params=grad_params
        )
        critic_loss = (
            jnp.square(q - target_q) * batch["valid"][..., -1]
        ).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "target_q_mean": target_q.mean(),
        }

    @partial(jax.jit, static_argnames=("flow_steps",))
    def adj_matching(self, obs, rng, flow_steps=None):
        """Original QAM adjoint-matching procedure."""
        flow_steps = (
            self.config["flow_steps"] if flow_steps is None else flow_steps
        )

        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"]
            if self.config["action_chunking"]
            else 1
        )
        x = jax.random.normal(
            rng, shape=obs.shape[:-1] + (action_dim,)
        )

        actor_slow = self.network.select(
            "target_actor_slow"
            if self.config["target_actor"]
            else "actor_slow"
        )

        h = 1.0 / flow_steps
        xs = [x]
        ts = []

        # Intentionally preserved from the supplied QAM implementation.
        for i, key in zip(
            range(flow_steps), jax.random.split(rng, flow_steps)
        ):
            t = (
                i / flow_steps
                * jnp.ones_like(x[..., 0:1])
            )
            sigma = jnp.sqrt(
                2.0 * (1.0 - t + h) / (t + h)
            )
            noise = jax.random.normal(key, x.shape)

            if i != flow_steps - 1:
                if self.config["residual"]:
                    v = (
                        self.network.select("actor_fast")(obs, x, t)
                        + actor_slow(obs, x, t)
                    )
                else:
                    v = self.network.select("actor_fast")(obs, x, t)

                x = (
                    x
                    + h * (2.0 * v - x / (t + h))
                    + jnp.sqrt(h) * sigma * noise
                )
            else:
                # ODE integration for the final step, as in QAM.
                x = x + h * actor_slow(obs, x, t)

            xs.append(x)
            ts.append(t)

        critic_network = (
            "target_critic"
            if self.config["use_target_grad"]
            else "critic"
        )

        if self.config["clip_adj"]:
            grad_fn = jax.grad(
                lambda x_obs, y_action: self.network.select(
                    critic_network
                )(
                    x_obs, jnp.clip(y_action, -1.0, 1.0)
                )
                .mean(axis=0)
                .sum(),
                1,
            )
        else:
            grad_fn = jax.grad(
                lambda x_obs, y_action: self.network.select(
                    critic_network
                )(x_obs, y_action)
                .mean(axis=0)
                .sum(),
                1,
            )

        # Exact QAM terminal condition.
        adj = (
            -grad_fn(obs, xs[-1])
            * self.config["inv_temp"]
        )

        pre_adj_info = {
            "adj_max": jnp.abs(adj).max(),
            "adj_std": jnp.abs(adj).std(),
            "adj_mean": jnp.abs(adj).mean(),
        }

        adjs = []
        for i in reversed(range(flow_steps)):
            t = (
                i / flow_steps
                * jnp.ones_like(x[..., 0:1])
            )

            def fn(xi):
                return (
                    2.0 * actor_slow(obs, xi, t + h)
                    - xi / (t + h)
                )

            vjp = jax.vjp(fn, xs[i])[1](adj)[0]
            adj = adj + h * vjp
            adjs.append(adj)

        return (
            jnp.stack(xs[:-1], axis=0),
            jnp.stack(list(reversed(adjs)), axis=0),
            jnp.stack(ts, axis=0),
            pre_adj_info,
        )

    def _geometric_expansion(
        self,
        batch,
        batch_actions,
        grad_params,
        subset_rng,
        sample_rng,
    ):
        """Build expanded slow-flow targets and the Gaussian actor loss.

        The Gaussian actor predicts absolute actions conditioned on
        (observation, behavior action). It is not a residual edit policy.
        """
        batch_size = batch_actions.shape[0]
        mixture_prob = float(
            self.config["geo_mixture_prob"]
        )
        num_geo = int(batch_size * mixture_prob)

        info = {
            "geo_num_augmented": jnp.asarray(
                num_geo, dtype=jnp.float32
            ),
            "geo_fraction_augmented": jnp.asarray(
                num_geo / batch_size, dtype=jnp.float32
            ),
        }

        if num_geo <= 0:
            return batch_actions, jnp.asarray(0.0), info

        # Random fixed-cardinality subset. This avoids relying on batch order.
        permutation = jax.random.permutation(
            subset_rng, batch_size
        )
        geo_indices = permutation[:num_geo]

        obs_geo = batch["observations"][geo_indices]
        actions_geo = batch_actions[geo_indices]

        # The dataset action is conditioning information only. The actor output
        # represents a complete absolute action.
        geo_input = jnp.concatenate(
            (obs_geo, actions_geo), axis=-1
        )
        geo_dist = self.network.select("geo_actor")(
            geo_input, params=grad_params
        )

        # Stochastic reparameterized sample: used only to train the generator.
        geo_raw_sample = geo_dist.sample(seed=sample_rng)
        geo_sample_action_unclipped = (
            self.config["geo_action_scale"]
            * geo_raw_sample
        )
        geo_sample_action = jnp.clip(
            geo_sample_action_unclipped, -1.0, 1.0
        )

        # Critic parameters are intentionally frozen here because no
        # `params=grad_params` argument is supplied. Gradients still propagate
        # through the sampled action into geo_actor.
        geo_qs = self.network.select("critic")(
            obs_geo, actions=geo_sample_action
        )
        geo_q_mean = geo_qs.mean(axis=0)
        geo_q_std = geo_qs.std(axis=0)
        geo_q_loss = -geo_q_mean.mean()

        geo_loss = geo_q_loss
        info.update(
            {
                "geo_q_loss": geo_q_loss,
                "geo_q_mean": geo_q_mean.mean(),
                "geo_q_std": geo_q_std.mean(),
            }
        )

        if self.config["geo_use_entropy"]:
            geo_log_prob = geo_dist.log_prob(
                geo_raw_sample
            )

            # Actor entropy term: alpha is frozen for this term.
            geo_alpha_frozen = self.network.select(
                "geo_alpha"
            )()
            geo_entropy_loss = (
                geo_alpha_frozen * geo_log_prob
            ).mean()

            # Temperature term: only alpha is optimized here.
            geo_entropy = -jax.lax.stop_gradient(
                geo_log_prob
            ).mean()
            geo_alpha = self.network.select(
                "geo_alpha"
            )(params=grad_params)
            geo_alpha_loss = (
                geo_alpha
                * (
                    geo_entropy
                    - self.config["geo_target_entropy"]
                )
            ).mean()

            geo_loss = (
                geo_loss
                + geo_entropy_loss
                + geo_alpha_loss
            )
            info.update(
                {
                    "geo_entropy": geo_entropy,
                    "geo_entropy_loss": geo_entropy_loss,
                    "geo_alpha_loss": geo_alpha_loss,
                    "geo_alpha": geo_alpha,
                    "geo_target_entropy": jnp.asarray(
                        self.config["geo_target_entropy"]
                    ),
                }
            )

        # Deterministic mode: used only as the slow-flow training target.
        if self.config["geo_use_mode_for_targets"]:
            geo_raw_target = geo_dist.mode()
        else:
            geo_raw_target = geo_raw_sample

        geo_target_unclipped = (
            self.config["geo_action_scale"]
            * geo_raw_target
        )
        geo_target = jnp.clip(
            geo_target_unclipped, -1.0, 1.0
        )
        geo_target = jax.lax.stop_gradient(
            geo_target
        )

        expanded_targets = batch_actions.at[
            geo_indices
        ].set(geo_target)

        data_qs = self.network.select("critic")(
            obs_geo, actions=actions_geo
        )
        data_q_mean = data_qs.mean(axis=0)

        info.update(
            {
                "geo_target_abs_mean": jnp.abs(
                    geo_target
                ).mean(),
                "geo_target_distance_l2": jnp.linalg.norm(
                    geo_target - actions_geo, axis=-1
                ).mean(),
                "geo_target_clip_fraction": jnp.mean(
                    jnp.abs(geo_target_unclipped) > 1.0
                ),
                "geo_sample_clip_fraction": jnp.mean(
                    jnp.abs(geo_sample_action_unclipped)
                    > 1.0
                ),
                "geo_predicted_q_improvement": (
                    geo_q_mean - data_q_mean
                ).mean(),
                "geo_data_q_mean": data_q_mean.mean(),
            }
        )

        return expanded_targets, geo_loss, info

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
        else:
            batch_actions = batch["actions"][..., 0, :]

        batch_size, action_dim = batch_actions.shape

        # Preserve the supplied QAM random streams exactly:
        # key 0 was previously unused after this split, so it is now used only
        # as the root for geometric-expansion randomness. x_rng, t_rng, and
        # adj_rng remain the same split positions as in QAM.
        geo_root_rng, x_rng, t_rng, adj_rng, _unused_rng = (
            jax.random.split(rng, 5)
        )
        geo_subset_rng, geo_sample_rng = (
            jax.random.split(geo_root_rng, 2)
        )

        x_0 = jax.random.normal(
            x_rng, (batch_size, action_dim)
        )

        x_1, geo_loss, geo_info = (
            self._geometric_expansion(
                batch=batch,
                batch_actions=batch_actions,
                grad_params=grad_params,
                subset_rng=geo_subset_rng,
                sample_rng=geo_sample_rng,
            )
        )

        t = jax.random.uniform(
            t_rng, (batch_size, 1)
        )
        x_t = (1.0 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_slow")(
            batch["observations"],
            x_t,
            t,
            params=grad_params,
        )
        flow_loss = jnp.mean(
            jnp.square(pred - vel).mean(axis=-1)
            * batch["valid"][..., -1]
        )
        actor_loss = flow_loss + geo_loss

        info = {
            "flow_loss": flow_loss,
            "geo_loss": geo_loss,
            **geo_info,
        }

        actor_slow = self.network.select(
            "target_actor_slow"
            if self.config["target_actor"]
            else "actor_slow"
        )

        xs, adjs, ts, pre_adj_info = self.adj_matching(
            batch["observations"], adj_rng
        )
        h = 1.0 / self.config["flow_steps"]
        sigmas = jnp.sqrt(
            2.0 * (1.0 - ts + h) / (ts + h)
        )

        observations = jnp.repeat(
            batch["observations"][None],
            self.config["flow_steps"],
            axis=0,
        )
        vf_fine = self.network.select(
            "actor_fast"
        )(
            observations,
            xs,
            ts,
            params=grad_params,
        )
        vf_base = actor_slow(
            observations, xs, ts
        )

        if self.config["residual"]:
            adj_loss = jnp.sum(
                jnp.square(
                    vf_fine * 2.0 / sigmas
                    + sigmas * adjs
                ),
                axis=-1,
            )
        else:
            adj_loss = jnp.sum(
                jnp.square(
                    (vf_fine - vf_base)
                    * 2.0
                    / sigmas
                    + sigmas * adjs
                ),
                axis=-1,
            )

        adj_loss = jnp.mean(
            jnp.sum(adj_loss, axis=0)
        )
        actor_loss = actor_loss + adj_loss

        info["adj_loss"] = adj_loss
        info["fast_loss"] = adj_loss
        info.update(pre_adj_info)

        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = self.rng if rng is None else rng
        rng, actor_rng, critic_rng = (
            jax.random.split(rng, 3)
        )

        critic_loss, critic_info = self.critic_loss(
            batch, grad_params, critic_rng
        )
        for key, value in critic_info.items():
            info[f"critic/{key}"] = value

        actor_loss, actor_info = self.actor_loss(
            batch, grad_params, actor_rng
        )
        for key, value in actor_info.items():
            info[f"actor/{key}"] = value

        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: (
                p * self.config["tau"]
                + tp * (1.0 - self.config["tau"])
            ),
            self.network.params[
                f"modules_{module_name}"
            ],
            self.network.params[
                f"modules_target_{module_name}"
            ],
        )
        network.params[
            f"modules_target_{module_name}"
        ] = new_target_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(
                batch, grad_params, rng=rng
            )

        new_network, info = (
            agent.network.apply_loss_fn(
                loss_fn=loss_fn
            )
        )
        agent.target_update(
            new_network, "critic"
        )
        agent.target_update(
            new_network, "actor_slow"
        )

        return (
            agent.replace(
                network=new_network, rng=new_rng
            ),
            info,
        )

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(
            self._update, self, batch
        )
        return agent, jax.tree_util.tree_map(
            lambda x: x.mean(), infos
        )

    @jax.jit
    def sample_actions(
        self, observations, rng
    ):
        # Preserve QAM's original key split even though this dedicated agent
        # has no inference-time edit policy.
        rng, _unused_edit_rng = jax.random.split(rng)

        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"]
            if self.config["action_chunking"]
            else 1
        )

        noises = jax.random.normal(
            rng,
            (
                *observations.shape[
                    : -len(self.config["ob_dims"])
                ],
                self.config["best_of_n"],
                action_dim,
            ),
        )
        observations = jnp.repeat(
            observations[..., None, :],
            self.config["best_of_n"],
            axis=-2,
        )

        if self.config["inv_temp"] == 0.0:
            actions = self.compute_flow_actions(
                observations,
                noises,
                model="slow",
            )
        else:
            actions = self.compute_flow_actions(
                observations,
                noises,
                model=(
                    "slow,fast"
                    if self.config["residual"]
                    else "fast"
                ),
            )

        actions = jnp.clip(
            actions, -1.0, 1.0
        )

        q = self.network.select("critic")(
            observations, actions
        ).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        batch_shape = indices.shape
        flat_indices = indices.reshape(-1)
        flat_batch_size = len(flat_indices)

        actions = jnp.reshape(
            actions,
            (
                -1,
                self.config["best_of_n"],
                action_dim,
            ),
        )[
            jnp.arange(flat_batch_size),
            flat_indices,
            :,
        ].reshape(batch_shape + (action_dim,))

        return actions

    @partial(
        jax.jit, static_argnames=("model",)
    )
    def compute_flow_actions(
        self,
        observations,
        noises,
        model="slow",
    ):
        actions = noises
        networks = [
            self.network.select(
                f"actor_{name}"
            )
            for name in model.split(",")
        ]

        for i in range(
            self.config["flow_steps"]
        ):
            t = jnp.full(
                (*observations.shape[:-1], 1),
                i / self.config["flow_steps"],
            )
            velocities = sum(
                network(
                    observations, actions, t
                )
                for network in networks
            )
            actions = (
                actions
                + velocities
                / self.config["flow_steps"]
            )

        return jnp.clip(
            actions, -1.0, 1.0
        )

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(
            rng, 2
        )

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]

        if config["action_chunking"]:
            full_actions = jnp.concatenate(
                [ex_actions]
                * config["horizon_length"],
                axis=-1,
            )
        else:
            full_actions = ex_actions

        full_action_dim = full_actions.shape[-1]

        if config["geo_target_entropy"] is None:
            config["geo_target_entropy"] = (
                -config[
                    "geo_target_entropy_multiplier"
                ]
                * full_action_dim
            )

        critic_def = Value(
            hidden_dims=config[
                "value_hidden_dims"
            ],
            layer_norm=config[
                "value_layer_norm"
            ],
            num_ensembles=config["num_qs"],
        )
        actor_def = ActorVectorField(
            hidden_dims=config[
                "actor_hidden_dims"
            ],
            layer_norm=config[
                "actor_layer_norm"
            ],
            action_dim=full_action_dim,
        )

        network_info = {
            "critic": (
                critic_def,
                (ex_observations, full_actions),
            ),
            "target_critic": (
                copy.deepcopy(critic_def),
                (ex_observations, full_actions),
            ),
            "actor_fast": (
                copy.deepcopy(actor_def),
                (
                    ex_observations,
                    full_actions,
                    ex_times,
                ),
            ),
            # Preserved from the supplied QAM module for architecture and
            # initialization parity, even though QAM does not update/use it
            # in the default non-residual configuration.
            "target_actor_fast": (
                copy.deepcopy(actor_def),
                (
                    ex_observations,
                    full_actions,
                    ex_times,
                ),
            ),
            "actor_slow": (
                copy.deepcopy(actor_def),
                (
                    ex_observations,
                    full_actions,
                    ex_times,
                ),
            ),
            "target_actor_slow": (
                copy.deepcopy(actor_def),
                (
                    ex_observations,
                    full_actions,
                    ex_times,
                ),
            ),
        }

        if config["geo_mixture_prob"] > 0.0:
            geo_actor_base_cls = partial(
                MLP,
                hidden_dims=config[
                    "geo_hidden_dims"
                ],
                activate_final=True,
            )
            geo_actor_def = TanhNormal(
                geo_actor_base_cls,
                full_action_dim,
            )
            geo_input = jnp.concatenate(
                (ex_observations, full_actions),
                axis=-1,
            )
            network_info["geo_actor"] = (
                geo_actor_def, geo_input
            )

            if config["geo_use_entropy"]:
                network_info["geo_alpha"] = (
                    LogParam(),
                    (),
                )

        networks = {
            key: value[0]
            for key, value in network_info.items()
        }
        network_args = {
            key: value[1]
            for key, value in network_info.items()
        }
        network_def = ModuleDict(networks)

        if config["clip_grad"]:
            network_tx = optax.chain(
                optax.clip_by_global_norm(
                    max_norm=1.0
                ),
                optax.adam(
                    learning_rate=config["lr"]
                ),
            )
        else:
            network_tx = optax.adam(
                learning_rate=config["lr"]
            )

        network_params = network_def.init(
            init_rng, **network_args
        )["params"]
        network = TrainState.create(
            network_def,
            network_params,
            tx=network_tx,
        )

        params = network.params
        params["modules_target_critic"] = (
            params["modules_critic"]
        )
        params["modules_target_actor_slow"] = (
            params["modules_actor_slow"]
        )

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim

        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(
                **config
            ),
        )


# Common dynamic-loader alias.
Agent = MEAMGeoAgent


def get_config():
    return ml_collections.ConfigDict(
        {
            "agent_name": "meam_geo",
            "ob_dims": (
                ml_collections.config_dict
                .placeholder(list)
            ),
            "action_dim": (
                ml_collections.config_dict
                .placeholder(int)
            ),

            # Shared QAM architecture and optimization.
            "lr": 3e-4,
            "batch_size": 256,
            "actor_hidden_dims": (
                512, 512, 512, 512
            ),
            "actor_layer_norm": False,
            "value_hidden_dims": (
                512, 512, 512, 512
            ),
            "value_layer_norm": True,

            # Q-chunking.
            "horizon_length": (
                ml_collections.config_dict
                .placeholder(int)
            ),
            "action_chunking": False,

            # QAM.
            "num_qs": 10,
            "rho": 0.5,
            "discount": 0.99,
            "tau": 0.005,
            "flow_steps": 10,
            "best_of_n": 1,
            "inv_temp": 0.3,
            "target_actor": True,
            "residual": False,
            "clip_adj": True,
            "clip_grad": True,
            "use_target_grad": True,

            # Gaussian geometric expansion.
            "geo_mixture_prob": 0.1,
            "geo_hidden_dims": (
                512, 512, 512, 512
            ),
            "geo_action_scale": 1.0,

            # The Gaussian is conditioned on (s, a_data) and predicts a full
            # absolute action. There is intentionally no misleading
            # "augment_as_edit" flag.
            "geo_use_mode_for_targets": True,

            # SAC-style entropy for the auxiliary expansion actor only.
            # None is resolved in create() to:
            #   -geo_target_entropy_multiplier * full_action_dim
            # With the default multiplier 0.5, this is -d_a / 2.
            "geo_use_entropy": True,
            "geo_target_entropy": None,
            "geo_target_entropy_multiplier": 0.5,
        }
    )
