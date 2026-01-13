import functools
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField

class AdjointMatchingAgent(flax.struct.PyTreeNode):
    """
    Adjoint Matching Agent for Finetuning.
    Uses a pre-trained Critic (from IQL/FQL) as the Reward Model.
    """
    rng: Any
    network: Any             # The Student Policy (Trainable)
    base_network: Any        # The Base Policy (Frozen, Source of Truth for BC)
    critic_agent: Any        # The Critic Agent (Frozen, Reward Model)
    config: Any = nonpytree_field()

    @classmethod
    def create(cls, 
               seed: int, 
               ex_observations: jnp.ndarray, 
               ex_actions: jnp.ndarray, 
               config: ml_collections.ConfigDict, 
               base_agent: Any, 
               critic_agent: Any):
        """
        Initializes the AM Agent by copying weights from the Base Agent.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # 1. Define Network Architecture
        network_def = ModuleDict({
            'student_policy': ActorVectorField(
                action_dim=config['action_dim'], 
                hidden_dims=config['actor_hidden_dims'], 
                layer_norm=config['actor_layer_norm']
            )
        })

        # 2. Extract Params from Base Agent
        params = base_agent.network.params
        if 'modules_actor_bc_flow' in params:
            base_params = params['modules_actor_bc_flow']
        elif 'modules_flow_actor' in params:
            base_params = params['modules_flow_actor']
        else:
            # Fallback for simple agents (like FlowBCAgent)
            base_params = params.get('actor_bc_flow', params)

        # 3. Initialize Student with Dummy Data
        network_tx = optax.adam(learning_rate=config['lr'])
        dummy_time = jnp.zeros((1, 1))
        
        # Initialize with correct key 'student_policy'
        init_params = network_def.init(init_rng, 
                                      student_policy=(ex_observations[:1], ex_actions[:1], dummy_time))['params']
        
        # 4. Overwrite Student Weights with Base Weights
        init_params['modules_student_policy'] = base_params
        
        network = TrainState.create(network_def, init_params, tx=network_tx)

        return cls(rng=rng, 
                   network=network, 
                   base_network=base_agent.network, 
                   critic_agent=critic_agent, 
                   config=flax.core.FrozenDict(**config))

    # --- Helpers ---

    def get_ode_drift(self, params, module_name, observations, actions, t_scalar):
        batch_size = actions.shape[0]
        times = jnp.full((batch_size, 1), t_scalar)
        # Use select() to properly handle TrainState wrapper
        return self.network.select(module_name)(observations, actions, times, params=params)
    
    def get_base_drift(self, observations, actions, t_scalar):
        batch_size = actions.shape[0]
        times = jnp.full((batch_size, 1), t_scalar)
        # Handle different base agent types
        if hasattr(self.base_network, 'select'):
             return self.base_network.select('actor_bc_flow')(observations, actions, times)
        else:
             # Fallback for simple TrainState without select (if used)
             return self.base_network.apply(
                {'params': self.base_network.params},
                observations=observations, actions=actions, times=times,
                method=lambda m: m['actor_bc_flow'](observations, actions, times)
            )

    # --- Core Logic: JAX Scans ---

    @functools.partial(jax.jit, static_argnames=('n_steps',))
    def forward_sde(self, rng, observations, n_steps, dt):
        batch_size = observations.shape[0]
        action_dim = self.config['action_dim']
        rng, init_rng = jax.random.split(rng)
        
        a_0 = jax.random.normal(init_rng, (batch_size, action_dim))
        
        def scan_step(carrier, i):
            a_t, current_rng = carrier
            t_float = i / n_steps
            t_safe = t_float + dt
            
            # Student Drift
            v_stud = self.get_ode_drift(self.network.params, 'student_policy', observations, a_t, t_float)
            
            # SDE Physics: 2*v - x/t
            drift = 2 * v_stud - (a_t / t_safe)
            sigma = jnp.sqrt(2 * (1 - t_float + dt) / (t_float + dt))
            
            current_rng, step_rng = jax.random.split(current_rng)
            noise = jax.random.normal(step_rng, a_t.shape)
            
            a_next = a_t + drift * dt + sigma * noise * jnp.sqrt(dt)
            return (a_next, current_rng), a_t

        _, traj_stacked = jax.lax.scan(scan_step, (a_0, rng), jnp.arange(n_steps))
        last_a = _[0]
        traj_full = jnp.concatenate([traj_stacked, last_a[None, ...]], axis=0)
        return traj_full

    @functools.partial(jax.jit, static_argnames=('n_steps',))
    def compute_targets(self, traj, observations, n_steps, dt):
        X_pre_final = traj[-2]
        t_pre_final = (n_steps - 1) / n_steps
        
        # Lookahead Step
        v_base_final = self.get_base_drift(observations, X_pre_final, t_pre_final)
        X_final_clean = X_pre_final + v_base_final * dt
        
        # Reward Gradient
        def reward_fn(a):
            q1, q2 = self.critic_agent.network.select('target_critic')(observations, actions=a)
            min_q = jnp.minimum(q1, q2)
            return jnp.sum(min_q) * self.config['reward_scale']

        grad_q = jax.grad(reward_fn)(X_final_clean)
        adjoint = -jnp.clip(grad_q, -self.config['q_grad_clip'], self.config['q_grad_clip'])
        
        avg_reward = reward_fn(X_final_clean) / (self.config['reward_scale'] * observations.shape[0])

        # Backward Scan
        def scan_backward(adjoint, args):
            i, a_curr = args
            t_float = i / n_steps
            t_safe = t_float + dt
            
            def drift_fn(a):
                return self.get_base_drift(observations, a, t_float)
            
            v_base, vjp_fn = jax.vjp(drift_fn, a_curr)
            
            term1_grads = vjp_fn(adjoint)[0]
            term2_grads = - (1.0 / t_safe) * adjoint
            vjp_total = jnp.clip(2 * term1_grads + term2_grads, -self.config['vjp_clip'], self.config['vjp_clip'])
            
            adjoint_next = adjoint + vjp_total * dt
            
            # [FIXED] Define sigma BEFORE calculating target_v
            sigma = jnp.sqrt(2 * (1 - t_float + dt) / (t_float + dt))
            target_v = v_base - (0.5 * sigma**2) * adjoint
            
            return adjoint_next, target_v

        indices = jnp.arange(n_steps)[::-1]
        traj_inputs = traj[:-1][::-1]
        _, targets = jax.lax.scan(scan_backward, adjoint, (indices, traj_inputs))
        
        return targets[::-1], avg_reward

    @jax.jit
    def update(self, batch):
        rng, step_rng = jax.random.split(self.rng)
        observations = batch['observations']
        n_steps = self.config['am_steps']
        dt = 1.0 / n_steps

        # 1. Forward SDE
        traj = self.forward_sde(step_rng, observations, n_steps, dt)
        
        # 2. Backward Targets
        targets, avg_rew = self.compute_targets(traj, observations, n_steps, dt)
        
        # 3. Update Student
        def loss_fn(params):
            def get_drift(i, x_t):
                return self.get_ode_drift(params, 'student_policy', observations, x_t, i / n_steps)
            
            preds = jax.vmap(get_drift)(jnp.arange(n_steps), traj[:-1])
            sq_err = jnp.sum((preds - targets)**2, axis=-1)
            loss = jnp.mean(jnp.clip(sq_err, a_max=self.config['LCT']))
            return loss, {'loss': loss}

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        info['avg_reward'] = avg_rew
        return self.replace(network=new_network, rng=rng), info

def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='adjoint_matching',
        lr=1e-4,
        batch_size=256,
        actor_hidden_dims=(512, 512, 512, 512),
        actor_layer_norm=False,
        am_steps=40,
        reward_scale=1.0, 
        LCT=10.0,
        q_grad_clip=10.0,
        vjp_clip=10.0,
        action_dim=ml_collections.config_dict.placeholder(int),
    ))
    return config