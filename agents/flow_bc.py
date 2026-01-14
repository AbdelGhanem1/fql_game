import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField

class FlowBCAgent(flax.struct.PyTreeNode):
    """
    Flow Matching Behavior Cloning Agent.
    Implements Conditional Flow Matching (CFM) / Rectified Flow.
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @jax.jit
    def compute_loss(self, batch, grad_params, rng):
        """
        Computes the Flow Matching MSE Loss.
        Objective: || v_theta(t, x_t) - (x_1 - x_0) ||^2
        """
        batch_size, action_dim = batch['actions'].shape
        
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        
        # 1. Sample Latent (Gaussian) and Target (Data)
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        
        # 2. Sample Time t ~ U[0, 1]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        
        # 3. Linear Interpolation (Rectified Flow / CFM)
        # This creates straight probability paths
        x_t = (1 - t) * x_0 + t * x_1
        target_vel = x_1 - x_0 

        # 4. Predict Velocity
        # Note: We assume the network handles the 'is_encoded' flag or we explicitly pass encoded obs
        pred_vel = self.network.select('actor_bc_flow')(
            batch['observations'], x_t, t, params=grad_params
        )
        
        # 5. MSE Loss
        loss = jnp.mean((pred_vel - target_vel) ** 2)

        return loss, {
            'loss': loss,
            'mse': loss,
        }

    @jax.jit
    def update(self, batch):
        """Performs one gradient update."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.compute_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None):
        """
        Inference: Solves the ODE dx/dt = v(t, x) using Euler method.
        """
        batch_size = observations.shape[0]
        action_dim = self.config['action_dim']
        
        if seed is None:
            seed = jax.random.PRNGKey(0) # Default if not provided
            
        # 1. Sample Noise x_0
        actions = jax.random.normal(seed, (batch_size, action_dim))
        
        # 2. Encode Observations (if encoder exists)
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)

        # 3. Euler Integration
        # We use a python loop for small steps (faster unroll) or lax.scan for many steps
        dt = 1.0 / self.config['flow_steps']
        
        def body_fn(i, val):
            curr_actions = val
            t = jnp.full((batch_size, 1), i * dt)
            
            # Predict velocity
            vel = self.network.select('actor_bc_flow')(
                observations, curr_actions, t, is_encoded=(self.config['encoder'] is not None)
            )
            return curr_actions + vel * dt

        # lax.fori_loop is cleaner for JIT compilation than python range
        actions = jax.lax.fori_loop(0, self.config['flow_steps'], body_fn, actions)
        
        # 4. Clip to valid action space [-1, 1]
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]

        # --- Encoders ---
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor_bc_flow'] = encoder_module()

        # --- Flow Network ---
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
        )

        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))

        # --- Initialize ---
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Update config with inferred dimensions
        config['action_dim'] = action_dim
        
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))

def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='flow_bc',
        lr=3e-4,
        batch_size=256,
        actor_hidden_dims=(512, 512, 512, 512),
        actor_layer_norm=False,
        flow_steps=10,
        encoder=ml_collections.config_dict.placeholder(str),
    ))
    return config