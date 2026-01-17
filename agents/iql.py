import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class IQLAgent(flax.struct.PyTreeNode):
    """Implicit Q-learning (IQL) agent with Ensemble Support."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        # [MODIFIED] Handle N ensembles for target
        qs = self.network.select('target_critic')(batch['observations'], actions=batch['actions'])
        
        # Take minimum across all ensemble members for conservative estimate
        # qs is a tuple/list, so we stack them
        if isinstance(qs, (tuple, list)):
            q = jnp.min(jnp.stack(qs, axis=0), axis=0)
        else:
            q = qs

        v = self.network.select('value')(batch['observations'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss for N ensembles."""
        next_v = self.network.select('value')(batch['next_observations'])
        q_target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        # [MODIFIED] Compute loss for all ensemble members
        qs = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        
        if isinstance(qs, (tuple, list)):
            critic_loss = sum(((q - q_target) ** 2).mean() for q in qs)
            q_mean_val = qs[0].mean() # Log the first one
            q_max_val = qs[0].max()
            q_min_val = qs[0].min()
        else:
            critic_loss = ((qs - q_target) ** 2).mean()
            q_mean_val = qs.mean()
            q_max_val = qs.max()
            q_min_val = qs.min()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q_mean_val,
            'q_max': q_max_val,
            'q_min': q_min_val,
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss (AWR or DDPG+BC)."""
        if self.config['actor_loss'] == 'awr':
            # AWR loss.
            v = self.network.select('value')(batch['observations'])
            
            # [MODIFIED] Use min of ensemble for advantage
            qs = self.network.select('critic')(batch['observations'], actions=batch['actions'])
            if isinstance(qs, (tuple, list)):
                q = jnp.min(jnp.stack(qs, axis=0), axis=0)
            else:
                q = qs
                
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }

            return actor_loss, actor_info
            
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            dist = self.network.select('actor')(batch['observations'], params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
                
            # [MODIFIED] Use min of ensemble
            qs = self.network.select('critic')(batch['observations'], actions=q_actions)
            if isinstance(qs, (tuple, list)):
                q = jnp.min(jnp.stack(qs, axis=0), axis=0)
            else:
                q = qs

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean())
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
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
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value'),
        )
        
        # [MODIFIED] INCREASED ENSEMBLES TO 10
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=10, 
            encoder=encoders.get('critic'),
        )
        
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
            encoder=encoders.get('actor'),
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        # --- [FIXED] DIAGNOSTIC: Check Ensemble Diversity ---
        try:
            # 1. Get Critic Params
            p_critic = network.params['modules_critic']
            
            # 2. Unwrap 'value_net' (created in Value.setup)
            if 'value_net' in p_critic:
                p_mlp = p_critic['value_net']
                
                # 3. Get First Layer (e.g., 'Dense_0')
                # Flax implicitly names layers Dense_0, Dense_1 unless named otherwise
                first_layer_key = sorted(list(p_mlp.keys()))[0] 
                
                # 4. Check Kernel Variance
                if 'kernel' in p_mlp[first_layer_key]:
                    kernel = p_mlp[first_layer_key]['kernel']
                    # kernel shape: (10, input_dim, output_dim) due to ensemblize
                    kernel_var = jnp.var(kernel, axis=0).mean()
                    print(f"DEBUG: Critic Ensemble Variance at Init: {kernel_var:.6f}")
                    
                    if kernel_var < 1e-6:
                        print("WARNING: ⚠️ Critics initialized identically! split_rngs might be missing.")
                else:
                    print(f"DEBUG: Could not find 'kernel' in {first_layer_key}")
            else:
                print("DEBUG: 'value_net' not found in critic params.")

        except Exception as e:
            print(f"DEBUG: Skipping variance check due to structure mismatch: {e}")
        # ----------------------------------------------------

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='iql',
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            actor_loss='awr',
            alpha=10.0,
            const_std=True,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config