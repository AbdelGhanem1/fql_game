import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

from functools import partial
from utils.networks import MLP, TanhNormal, LogParam

class MEAMAgent(flax.struct.PyTreeNode):
    """Q-learning with adjoint matching, mixture of priors, and entropy maximization."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    
    def critic_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=rng)
        next_actions = jnp.clip(next_actions, -1, 1)
        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)
        
        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        total_loss = critic_loss
        return total_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }
    
    @partial(jax.jit, static_argnames=("flow_steps"))
    def adj_matching(self, obs, rng, flow_steps=None):
        flow_steps = self.config["flow_steps"] if flow_steps is None else flow_steps

        action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        x = jax.random.normal(rng, shape=obs.shape[:-1] + (action_dim,))

        actor_slow = self.network.select("target_actor_slow" if self.config["target_actor"] else "actor_slow")

        h = 1 / flow_steps
        xs = [x]
        ts = []
        for i, key in zip(range(flow_steps), jax.random.split(rng, flow_steps)):
            t = i / flow_steps * jnp.ones_like(x[..., 0:1])
            sigma = jnp.sqrt(2 * (1 - t + h) / (t + h))
            noise = jax.random.normal(key, x.shape)
            if i != flow_steps - 1:
                if self.config["residual"]:
                    v = self.network.select("actor_fast")(obs, x, t) + actor_slow(obs, x, t)
                else:
                    v = self.network.select("actor_fast")(obs, x, t)
                x = x + h * (2 * v - x / (t + h)) + jnp.sqrt(h) * sigma * noise
            else:  # use ODE integration for the last step following the adjoint-matching paper
                x = x + h * actor_slow(obs, x, t)

            xs.append(x)
            ts.append(t)

        # Compute the critic's action gradient as the adjoint state initialization
        critic_network = "target_critic" if self.config["use_target_grad"] else "critic"
        
        # Kept the exact QAM clip logic here
        if self.config['clip_adj']:
            grad_fn = jax.grad(lambda x, y: self.network.select(critic_network)(x, jnp.clip(y, -1., 1.)).mean(axis=0).sum(), 1)
        else:
            grad_fn = jax.grad(lambda x, y: self.network.select(critic_network)(x, y).mean(axis=0).sum(), 1)

        q_grad = grad_fn(obs, xs[-1])
        q_grad_norm = jnp.linalg.norm(q_grad, axis=-1, keepdims=True)
        q_grad_norm_val = q_grad_norm.mean()

        # Added MEAM q-gradient clipping
        if self.config.get("clip_q_grad", True):
            q_clip_max = self.config.get("q_grad_clip_max", 10.0)
            q_clip_scale = jnp.minimum(1.0, q_clip_max / (q_grad_norm + 1.e-6))
            q_grad_safe = q_grad * q_clip_scale
            q_term = q_grad_safe * self.config["inv_temp"]
        else:
            q_term = q_grad * self.config["inv_temp"]

        # Added MEAM score net / alpha logic
        score_norm_val = 0.0
        if self.config["me_am_alpha"] > 0.:
            score_est = self.network.select("score_net")(obs, jnp.clip(xs[-1], -1., 1.))
            score_norm = jnp.linalg.norm(score_est, axis=-1, keepdims=True)
            score_norm_val = score_norm.mean()
            
            if self.config.get("clip_score", True):
                score_clip_max = self.config.get("score_clip_max", 1.0)
                score_clip_scale = jnp.minimum(1.0, score_clip_max / (score_norm + 1.e-6))
                score_est_safe = score_est * score_clip_scale
                score_term = self.config["me_am_alpha"] * score_est_safe
            else:
                score_term = self.config["me_am_alpha"] * score_est
            
            total_grad = q_term - score_term
        else:
            total_grad = q_term

        adj = -total_grad
        
        pre_adj_info = {
            "adj_max": jnp.abs(adj).max(),
            "adj_std": jnp.abs(adj).std(),
            "adj_mean": jnp.abs(adj).mean(),
            "q_grad_norm": q_grad_norm_val,
            "score_norm": score_norm_val,
        }
        
        adjs = []
        for i in reversed(range(flow_steps)):
            t = (i / flow_steps) * jnp.ones_like(x[..., 0:1])

            def fn(xi):
                return 2 * actor_slow(obs, xi, t + h) - xi / (t + h)
            
            vjp = jax.vjp(fn, xs[i])[1](adj)[0]
            adj = adj + h * vjp
            
            adjs.append(adj)
        return jnp.stack(xs[:-1], axis=0), jnp.stack(list(reversed(adjs)), axis=0), jnp.stack(ts, axis=0), pre_adj_info

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        
        batch_size, full_action_dim = batch_actions.shape
        
        # Extended split to include aug_rng for our augmentations
        rng, x_rng, t_rng, adj_rng, edit_rng, mix_mask_rng, aug_rng, score_rng = jax.random.split(rng, 8)

        ## BC flow-matching loss.
        x_0 = jax.random.normal(x_rng, (batch_size, full_action_dim))
        
        # --- Additive Temporally Coherent/Incoherent Augmentation ---
        if self.config["mixture_prob"] > 0.0:
            base_action_dim = self.config['action_dim']
            horizon = self.config['horizon_length'] if self.config["action_chunking"] else 1
            
            mix_type = self.config.get("mixture_prob_type", "ou")
            
            if mix_type == "ou":
                # 1. Ornstein-Uhlenbeck (Temporally correlated random walk)
                theta = self.config.get("ou_theta", 0.15)
                sigma = self.config.get("ou_sigma", 0.2)
                
                def ou_step(x_prev, key):
                    noise = jax.random.normal(key, (batch_size, base_action_dim))
                    x_new = x_prev * (1.0 - theta) + sigma * noise
                    return x_new, x_new
                    
                ou_keys = jax.random.split(aug_rng, horizon)
                init_state = jnp.zeros((batch_size, base_action_dim))
                _, generated_noise = jax.lax.scan(ou_step, init_state, ou_keys)
                
                # Reshape to match flattened batch_actions
                generated_noise = jnp.transpose(generated_noise, (1, 0, 2)).reshape((batch_size, -1))
                
            elif mix_type == "uniform":
                # 2. Correlated Uniform (Rigid trajectory shift)
                margin = self.config.get("mixture_margin", 0.5)
                
                # Sample ONCE per chunk: shape (batch_size, 1, base_action_dim)
                base_noise = jax.random.uniform(aug_rng, (batch_size, 1, base_action_dim), 
                                                minval=-margin, maxval=margin)
                
                # Broadcast across the horizon and flatten
                generated_noise = jnp.repeat(base_noise, horizon, axis=1).reshape((batch_size, -1))

            elif mix_type == "independent_uniform":
                # 3. Independent Uniform (High-frequency noise across all timesteps)
                margin = self.config.get("mixture_margin", 0.5)
                
                # Sample INDEPENDENTLY for every timestep: shape (batch_size, horizon, base_action_dim)
                generated_noise = jax.random.uniform(aug_rng, (batch_size, horizon, base_action_dim), 
                                                minval=-margin, maxval=margin)
                
                # Flatten
                generated_noise = generated_noise.reshape((batch_size, -1))
                
            elif mix_type == "gaussian":
                # 4. Correlated Gaussian (Rigid trajectory shift via standard normal)
                sigma = self.config.get("mixture_sigma", 0.2)
                
                # Sample ONCE per chunk: shape (batch_size, 1, base_action_dim)
                base_noise = jax.random.normal(aug_rng, (batch_size, 1, base_action_dim)) * sigma
                
                # Broadcast across the horizon and flatten
                generated_noise = jnp.repeat(base_noise, horizon, axis=1).reshape((batch_size, -1))
                
            else:
                generated_noise = jnp.zeros_like(batch_actions)
            
            # Additive augmentation with clipping to maintain valid action bounds
            augmented_actions = jnp.clip(batch_actions + generated_noise, -1.0, 1.0)
            
            # Apply the mixture mask
            mixture_mask = jax.random.bernoulli(mix_mask_rng, p=self.config["mixture_prob"], shape=(batch_size, 1)).astype(jnp.float32)
            x_1 = (1.0 - mixture_mask) * batch_actions + mixture_mask * augmented_actions
        else:
            x_1 = batch_actions

        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_slow')(batch['observations'], x_t, t, params=grad_params)
        flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * batch["valid"][..., -1])
        actor_loss = flow_loss
        
        info = {}
        total_fast_loss = 0
        actor_slow = self.network.select("target_actor_slow" if self.config["target_actor"] else "actor_slow")
        
        ## Adjoint-matching
        # Compute the adjoint states
        xs, adjs, ts, pre_adj_info = self.adj_matching(batch["observations"], adj_rng)
        
        # === SCORE NET TRAINING (DSM) ===
        if self.config["me_am_alpha"] > 0.0:
            if self.config["score_mode"] == "slow":
                model_for_score = "slow"
            else: 
                model_for_score = "slow,fast" if self.config["residual"] else "fast"

            flow_rng, eps_rng = jax.random.split(score_rng)

            flow_actions = self.compute_flow_actions(
                batch["observations"], 
                jax.random.normal(flow_rng, (batch_size, full_action_dim)),
                model=model_for_score,
                use_target=True 
            )
            
            sigma_score = self.config["score_sigma"]
            epsilon = jax.random.normal(eps_rng, flow_actions.shape)
            perturbed_actions = flow_actions + sigma_score * epsilon
            
            estimated_score = self.network.select('score_net')(batch["observations"], perturbed_actions, params=grad_params)
            target_score = -epsilon / sigma_score
            
            # Score Masking Fix: Mask out dummy padding transitions
            score_error = jnp.square(estimated_score - target_score).mean(axis=-1)
            score_loss = jnp.mean(score_error * batch["valid"][..., -1])
            
            actor_loss += score_loss
            info["score_loss"] = score_loss

        h = 1 / self.config["flow_steps"]
        sigmas = jnp.sqrt(2 * (1 - ts + h) / (ts + h))

        observations = jnp.repeat(batch["observations"][None], self.config["flow_steps"], axis=0)
        vf_fine = self.network.select("actor_fast")(observations, xs, ts, params=grad_params)

        vf_base = actor_slow(observations, xs, ts)
        
        # Compute the adjoint matching loss
        if self.config["residual"]:
            adj_loss = jnp.sum(jnp.square(vf_fine * 2 / sigmas + sigmas * adjs), axis=-1)
        else:
            adj_loss = jnp.sum(jnp.square((vf_fine - vf_base) * 2 / sigmas + sigmas * adjs), axis=-1)

        adj_loss = jnp.mean(jnp.sum(adj_loss, axis=0))

        info["adj_loss"] = adj_loss
        info.update(pre_adj_info)
        total_fast_loss += adj_loss

        if self.config["fql_alpha"] > 0.:
            edit_base_rng, edit_rng = jax.random.split(edit_rng, 2)
            fql_noises = jax.random.normal(edit_base_rng, (batch_size, full_action_dim))
            flow_actions = self.compute_flow_actions(batch["observations"], 
                fql_noises, 
                model="slow,fast" if self.config["residual"] else "fast")
            
            os_actions = self.network.select('one_step_actor')(
                batch["observations"], fql_noises, 
                params=grad_params)
            fql_distill_loss = jnp.mean((flow_actions - os_actions) ** 2)
            
            # FQL loss.
            os_actions = jnp.clip(os_actions, -1, 1)
            fql_qs = self.network.select(f'critic')(batch['observations'], actions=os_actions)
            fql_q = jnp.mean(fql_qs, axis=0)
            fql_q_loss = -fql_q.mean()

            info["fql_distill_loss"] = fql_distill_loss
            info["fql_q_loss"] = fql_q_loss

            actor_loss += fql_q_loss + fql_distill_loss * self.config["fql_alpha"]

        if self.config["edit_scale"] > 0.:
            edit_base_rng, edit_rng = jax.random.split(edit_rng, 2)
            flow_actions = self.compute_flow_actions(batch["observations"], 
                jax.random.normal(edit_base_rng, (batch_size, full_action_dim)), 
                model="slow,fast" if self.config["residual"] else "fast")
            
            edit_dist = self.network.select('edit_actor')(
                jnp.concatenate((batch["observations"], flow_actions), axis=-1), 
                params=grad_params)
            edit = edit_dist.sample(seed=edit_rng)
            edit_log_probs = edit_dist.log_prob(edit)
            
            edited_actions = flow_actions + edit * self.config["edit_scale"]
            
            # Edit policy loss.
            edited_actions = jnp.clip(edited_actions, -1, 1)
            qs = self.network.select(f'critic')(batch['observations'], actions=edited_actions)
            q = jnp.mean(qs, axis=0)
            edit_q_loss = -q.mean()

            edit_entropy_loss = (edit_log_probs * self.network.select('edit_alpha')()).mean()

            alpha = self.network.select('edit_alpha')(params=grad_params)
            entropy = -jax.lax.stop_gradient(edit_log_probs).mean()
            edit_alpha_loss = (alpha * (entropy - self.config['edit_target_entropy'])).mean()

            actor_loss += edit_q_loss + edit_entropy_loss + edit_alpha_loss

            info["edit_q_loss"] = edit_q_loss
            info["edit_entropy_loss"] = edit_entropy_loss
            info["edit_alpha_loss"] = edit_alpha_loss
            info["edit_entropy"] = entropy
            info["edit_alpha"] = alpha

        return actor_loss + total_fast_loss, {'flow_loss': flow_loss, "fast_loss": total_fast_loss, **info}
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        agent.target_update(new_network, 'actor_slow')

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng,
    ):
        rng, edit_rng = jax.random.split(rng)
        
        action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                self.config["best_of_n"], action_dim
            ),
        )
        observations = jnp.repeat(observations[..., None, :], self.config["best_of_n"], axis=-2)

        if self.config["fql_alpha"] > 0.:  # if fql_alpha > 0, use the one-step policy
            actions = self.network.select('one_step_actor')(
                observations, noises)
            actions = jnp.clip(actions, -1, 1)
        else:   # otherwise use the flow policy
            if self.config["inv_temp"] == 0.:
                actions = self.compute_flow_actions(observations, noises, model="slow")
            else:
                actions = self.compute_flow_actions(observations, noises, model="slow,fast" if self.config["residual"] else "fast")
            if self.config["edit_scale"] > 0.:  # if there is an edit policy, refine the action further
                edit_dist = self.network.select("edit_actor")(jnp.concatenate((observations, actions), axis=-1))
                actions = actions + edit_dist.sample(seed=edit_rng) * self.config["edit_scale"]
            actions = jnp.clip(actions, -1, 1)
        
        # best-of-n sampling
        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (action_dim,))

        return actions

    @partial(jax.jit, static_argnames=["model", "use_target"])
    def compute_flow_actions(
        self,
        observations,
        noises,
        model="slow",
        use_target=False,
    ):
        actions = noises
        prefix = "target_actor" if use_target else "actor"
        networks = [self.network.select(f'{prefix}_{m}') for m in model.split(",")]

        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = sum([network(observations, actions, t) for network in networks])
            
            actions = actions + vels / self.config['flow_steps']

        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        if config['edit_target_entropy'] is None:
            config['edit_target_entropy'] = -config['edit_target_entropy_multiplier'] * full_action_dim

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=config['num_qs'],
        )
        actor_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            layer_norm=config['actor_layer_norm'],
            action_dim=full_action_dim,
        )
        
        # === SCORE NET DEFINITION ===
        score_net_def_mlp = MLP(hidden_dims=tuple(list(config['score_net_hidden_dims']) + [full_action_dim]), activate_final=False, layer_norm=True)
        
        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_fast=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            target_actor_slow=(copy.deepcopy(actor_def), (ex_observations, full_actions, ex_times)),
            score_net=(score_net_def_mlp, (ex_observations, full_actions)),
        )

        assert (config["fql_alpha"] * config["edit_scale"] == 0.), "Only one of fql_alpha and edit_scale can be non-zero."
        
        if config["fql_alpha"] > 0.:
            network_info.update(dict(
                one_step_actor=(copy.deepcopy(actor_def), (ex_observations, full_actions, None)),
            ))

        if config["edit_scale"] > 0.:
            edit_actor_base_cls = partial(MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True)
            edit_actor_def = TanhNormal(edit_actor_base_cls, full_action_dim)
            alpha_def = LogParam()

            network_info.update(dict(
                edit_actor=(edit_actor_def, jnp.concatenate((ex_observations, full_actions), axis=-1)),
                edit_alpha=(alpha_def, ()),
            ))
        
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)

        if config["clip_grad"]:
            network_tx = optax.chain(optax.clip_by_global_norm(max_norm=1.0),  # clip grad norm
                optax.adam(learning_rate=config["lr"]))
        else:
            network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor_slow'] = params['modules_actor_slow']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='meam',  
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            
            ## Common hyperparameters
            lr=3e-4,  
            batch_size=256,  
            actor_hidden_dims=(512, 512, 512, 512),  
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),  
            value_layer_norm=True,
            
            ## Q-chunking hyperparameters
            horizon_length=ml_collections.config_dict.placeholder(int), 
            action_chunking=False,                                      
            
            ## RL hyperparameters
            num_qs=10,       
            rho=0.5,        

            discount=0.99,  
            tau=0.005,      
            flow_steps=10,  

            best_of_n=1,    
            
            ## Main hyperparameter(s)
            inv_temp=0.3,   
            fql_alpha=0.,   
            edit_scale=0.,  
            
            ## MEAM Components 
        
            score_net_hidden_dims=(256, 256),
            score_sigma=0.1,           
            score_mode='slow',         
            clip_q_grad=False,
            q_grad_clip_max=10.0,
            clip_score=False,
            score_clip_max=1.0,
    
            me_am_alpha=0.0,            
            mixture_prob=0.1,           
            mixture_prob_type='independent_uniform', # 'ou', 'uniform', or 'gaussian'
            mixture_margin=0.5,          # Used for 'uniform'
            mixture_sigma=0.2,           # Used for 'gaussian'
            ou_theta=0.15,               # Used for 'ou'
            ou_sigma=0.2,                # Used for 'ou'
            
            ## Other variants/hyperparameter(s)
            target_actor=True,
            residual=False,
            clip_adj=True,
            clip_grad=True,
            use_target_grad=True,
            edit_target_entropy=ml_collections.config_dict.placeholder(float),  
            edit_target_entropy_multiplier=0.5,  
        )
    )
    return config
