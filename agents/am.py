import functools
from typing import Any, Dict, Tuple

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

        # 1. Define Network Architecture (Same as FQL)
        # We reuse ActorVectorField to ensure compatibility and robust init
        network_def = ModuleDict({
            'student_policy': ActorVectorField(
                action_dim=config['action_dim'], 
                hidden_dims=config['actor_hidden_dims'], 
                layer_norm=config['actor_layer_norm']
            )
        })

        # 2. Extract Params from Base Agent
        # We assume base_agent is an FQLAgent or IQLAgent with a flow actor.
        # Adjust 'modules_actor_bc_flow' if your base agent uses a different name.
        if 'modules_actor_bc_flow' in base_agent.network.params:
            base_params = base_agent.network.params['modules_actor_bc_flow']
        elif 'modules_flow_actor' in base_agent.network.params:
            base_params = base_agent.network.params['modules_flow_actor']
        else:
            raise ValueError("Could not find flow params in Base Agent. Check param keys.")

        # 3. Initialize Student with Dummy Data
        network_tx = optax.adam(learning_rate=config['lr'])
        dummy_obs = ex_observations[:1]
        dummy_acts = ex_actions[:1]
        dummy_time = jnp.zeros((1, 1))
        
        init_params = network_def.init(init_rng, 
                                      observations=dummy_obs, 
                                      actions=dummy_acts, 
                                      times=dummy_time)['params']
        
        # 4. Overwrite Student Weights with Base Weights
        # This ensures we start exactly where BC left off (The "Blob")
        init_params['modules_student_policy'] = base_params
        
        network = TrainState.create(network_def, init_params, tx=network_tx)

        return cls(rng=rng, 
                   network=network, 
                   base_network=base_agent.network, 
                   critic_agent=critic_agent, 
                   config=flax.core.FrozenDict(**config))

    # --- Core Logic: Helper Wrappers ---

    def get_ode_drift(self, params, module_name, observations, actions, t_scalar):
        """Calculates v(t, x) for a batch."""
        batch_size = actions.shape[0]
        times = jnp.full((batch_size, 1), t_scalar)
        
        # Call the specific module (Student or Base)
        # utils.networks.ActorVectorField.__call__ takes (observations, actions, times)
        return self.network.apply(
            {'params': params},
            observations=observations,
            actions=actions,
            times=times,
            method=lambda module: module[module_name](observations, actions, times)
        )
    
    def get_base_drift(self, observations, actions, t_scalar):
        """Drift from the Frozen Base Policy."""
        batch_size = actions.shape[0]
        times = jnp.full((batch_size, 1), t_scalar)
        # Use the base network's 'select' method if available, or direct apply
        return self.base_network.select('actor_bc_flow')(observations, actions, times)

    # --- Core Logic: JAX Scans (Validated in Toy Env) ---

    @functools.partial(jax.jit, static_argnames=('n_steps',))
    def forward_sde(self, rng, observations, n_steps, dt):
        """Generates training trajectories with SDE noise."""
        batch_size = observations.shape[0]
        action_dim = self.config['action_dim']
        rng, init_rng = jax.random.split(rng)
        
        a_0 = jax.random.normal(init_rng, (batch_size, action_dim))
        
        def scan_step(carrier, i):
            a_t, current_rng = carrier
            t_float = i / n_steps
            t_safe = t_float + dt
            
            # 1. Student Drift
            v_stud = self.get_ode_drift(self.network.params, 'modules_student_policy', observations, a_t, t_float)
            
            # 2. SDE Physics: 2*v - x/t
            drift = 2 * v_stud - (a_t / t_safe)
            sigma = jnp.sqrt(2 * (1 - t_float + dt) / (t_float + dt))
            
            current_rng, step_rng = jax.random.split(current_rng)
            noise = jax.random.normal(step_rng, a_t.shape)
            
            a_next = a_t + drift * dt + sigma * noise * jnp.sqrt(dt)
            return (a_next, current_rng), a_t

        _, traj_stacked = jax.lax.scan(scan_step, (a_0, rng), jnp.arange(n_steps))
        
        # Append final state for targets
        last_a = _[0]
        traj_full = jnp.concatenate([traj_stacked, last_a[None, ...]], axis=0)
        return traj_full

    @functools.partial(jax.jit, static_argnames=('n_steps',))
    def compute_targets(self, traj, observations, n_steps, dt):
        """Computes Adjoint Matching Regression Targets."""
        X_pre_final = traj[-2]
        t_pre_final = (n_steps - 1) / n_steps
        
        # 1. Lookahead Step (Denoising at t=1)
        v_base_final = self.get_base_drift(observations, X_pre_final, t_pre_final)
        X_final_clean = X_pre_final + v_base_final * dt
        
        # 2. Compute Reward Gradient (Adjoint at t=1)
        def reward_fn(a):
            # Access the Frozen Critic to get Q-values
            # We assume critic_agent has a method to get min(Q1, Q2) or similar
            # If using FQL/IQL structure:
            q1, q2 = self.critic_agent.network.select('target_critic')(observations, actions=a)
            min_q = jnp.minimum(q1, q2)
            return jnp.sum(min_q) * self.config['reward_scale']

        grad_q = jax.grad(reward_fn)(X_final_clean)
        
        # Clip Gradients (Safety)
        adjoint = -grad_q
        adjoint = jnp.clip(adjoint, -self.config['q_grad_clip'], self.config['q_grad_clip'])
        
        avg_reward = reward_fn(X_final_clean) / (self.config['reward_scale'] * observations.shape[0])

        # 3. Backward Scan
        def scan_backward(adjoint, args):
            i, a_curr = args
            t_float = i / n_steps
            t_safe = t_float + dt
            
            # VJP of Base Drift
            def drift_fn(a):
                return self.get_base_drift(observations, a, t_float)
            
            v_base, vjp_fn = jax.vjp(drift_fn, a_curr)
            
            # Adjoint Dynamics
            term1_grads = vjp_fn(adjoint)[0]
            term2_grads = - (1.0 / t_safe) * adjoint
            vjp_total = 2 * term1_grads + term2_grads
            vjp_total = jnp.clip(vjp_total, -self.config['vjp_clip'], self.config['vjp_clip'])
            
            adjoint_next = adjoint + vjp_total * dt
            
            # Regression Target
            sigma = jnp.sqrt(2 * (1 - t_float + dt) / (t_float + dt))
            target_v = v_base - (0.5 * sigma**2) * adjoint
            
            return adjoint_next, target_v

        indices = jnp.arange(n_steps)[::-1]
        traj_inputs = traj[:-1][::-1]
        _, targets = jax.lax.scan(scan_backward, adjoint, (indices, traj_inputs))
        
        return targets[::-1], avg_reward

    @jax.jit
    def update(self, batch):
        """The Main Training Step."""
        rng, step_rng = jax.random.split(self.rng)
        
        observations = batch['observations']
        n_steps = self.config['am_steps']
        dt = 1.0 / n_steps

        # A. Generate Trajectory (SDE)
        traj = self.forward_sde(step_rng, observations, n_steps, dt)
        
        # B. Calculate Targets (Adjoint)
        targets, avg_rew = self.compute_targets(traj, observations, n_steps, dt)
        
        # C. Update Student (MSE Loss)
        def loss_fn(params):
            def get_drift(i, x_t):
                return self.get_ode_drift(params, 'modules_student_policy', observations, x_t, i / n_steps)
            
            # Calculate drift for all steps in parallel (vmap)
            preds = jax.vmap(get_drift)(jnp.arange(n_steps), traj[:-1])
            
            sq_err = jnp.sum((preds - targets)**2, axis=-1)
            loss_clipped = jnp.clip(sq_err, a_max=self.config['LCT'])
            return jnp.mean(loss_clipped)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        
        info['avg_reward'] = avg_reward
        new_agent = self.replace(network=new_network, rng=rng)
        return new_agent, info

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