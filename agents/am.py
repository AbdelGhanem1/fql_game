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
    Adjoint Matching Agent for Finetuning with Robust LCB Guidance.
    """
    rng: Any
    network: Any             
    base_network: Any        
    critic_agent: Any        
    config: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, base_agent, critic_agent):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        network_def = ModuleDict({
            'student_policy': ActorVectorField(
                action_dim=config['action_dim'], 
                hidden_dims=config['actor_hidden_dims'], 
                layer_norm=config['actor_layer_norm']
            )
        })

        # --- Extract Base Parameters ---
        params = base_agent.network.params
        if 'modules_actor_bc_flow' in params:
            base_params = params['modules_actor_bc_flow']
        elif 'modules_flow_actor' in params:
            base_params = params['modules_flow_actor']
        else:
            base_params = params.get('actor_bc_flow', params)

        network_tx = optax.adam(learning_rate=config['lr'])
        dummy_time = jnp.zeros((1, 1))
        
        # --- Initialize Student Parameters ---
        init_params = network_def.init(init_rng, 
                                      student_policy=(ex_observations[:1], ex_actions[:1], dummy_time))['params']
        
        # --- Inject Base Weights ---
        print("\n=== ADJOINT MATCHING INITIALIZATION ===")
        if 'student_policy' in init_params:
            init_params['student_policy'] = base_params
            print(f"✅ Key Match: Injected weights into 'student_policy'.")
        elif 'modules_student_policy' in init_params:
            init_params['modules_student_policy'] = base_params
            print(f"✅ Key Match: Injected weights into 'modules_student_policy'.")
        else:
            raise ValueError(f"❌ FATAL ERROR: Could not find target key. Keys: {list(init_params.keys())}")

        network = TrainState.create(network_def, init_params, tx=network_tx)

        return cls(rng=rng, 
                   network=network, 
                   base_network=base_agent.network, 
                   critic_agent=critic_agent, 
                   config=flax.core.FrozenDict(**config))

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        is_single_input = False
        if observations.ndim == 1:
            is_single_input = True
            observations = observations[None, :]

        batch_size = observations.shape[0]
        action_dim = self.config['action_dim']
        
        if seed is None:
            seed = jax.random.PRNGKey(0)
            
        actions = jax.random.normal(seed, (batch_size, action_dim)) * temperature
        
        # Use ode_steps
        steps = self.config.get('ode_steps', 20) 
        dt = 1.0 / steps
        
        def body_fn(i, val):
            curr_actions = val
            t = jnp.full((batch_size, 1), i * dt)
            vel = self.network.select('student_policy')(
                observations, curr_actions, t, params=self.network.params
            )
            return curr_actions + vel * dt

        actions = jax.lax.fori_loop(0, steps, body_fn, actions)
        actions = jnp.clip(actions, -1, 1)

        if is_single_input:
            actions = actions[0]

        return actions

    def get_ode_drift(self, params, module_name, observations, actions, t_scalar):
        batch_size = actions.shape[0]
        times = jnp.full((batch_size, 1), t_scalar)
        return self.network.select(module_name)(observations, actions, times, params=params)
    
    def get_base_drift(self, observations, actions, t_scalar):
        batch_size = actions.shape[0]
        times = jnp.full((batch_size, 1), t_scalar)
        if hasattr(self.base_network, 'select'):
             return self.base_network.select('actor_bc_flow')(observations, actions, times)
        else:
             return self.base_network.apply(
                {'params': self.base_network.params},
                observations=observations, actions=actions, times=times,
                method=lambda m: m['actor_bc_flow'](observations, actions, times)
            )

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
            
            v_stud = self.get_ode_drift(self.network.params, 'student_policy', observations, a_t, t_float)
            
            # Memoryless Drift
            drift = 2 * v_stud - (a_t / t_safe)
            sigma = jnp.sqrt(2 * (1 - t_float + dt/4) / (t_float + dt/4))
            
            current_rng, step_rng = jax.random.split(current_rng)
            noise = jax.random.normal(step_rng, a_t.shape)
            
            a_next = a_t + drift * dt + sigma * noise * jnp.sqrt(dt)
            return (a_next, current_rng), a_t

        _, traj_stacked = jax.lax.scan(scan_step, (a_0, rng), jnp.arange(n_steps))
        last_a = _[0]
        traj_full = jnp.concatenate([traj_stacked, last_a[None, ...]], axis=0)
        return traj_full, None

    @functools.partial(jax.jit, static_argnames=('n_steps',))
    def compute_targets(self, traj, observations, n_steps, dt):
        X_pre_final = traj[-2]
        t_pre_final = (n_steps - 1) / n_steps
        
        v_base_final = self.get_base_drift(observations, X_pre_final, t_pre_final)
        X_final_clean = X_pre_final + v_base_final * dt
        
        def reward_fn(a):
            a_clipped = jnp.clip(a, -1.0, 1.0)
            qs = self.critic_agent.network.select('target_critic')(observations, actions=a_clipped)
            
            # [FIXED] Handle Ensemble Shapes Correctly
            if isinstance(qs, (list, tuple)):
                qs_stack = jnp.stack(qs, axis=0)
            elif qs.ndim > 1 and qs.shape[0] < 30: 
                qs_stack = qs
            else:
                qs_stack = qs[None, ...] 

            q_mean = jnp.mean(qs_stack, axis=0)
            q_var = jnp.var(qs_stack, axis=0)
            q_std = jnp.sqrt(q_var + 1e-6)
            
            beta = self.config.get('uncertainty_beta', 2.0)
            robust_q = q_mean - beta * q_std
            
            return jnp.sum(robust_q) / (self.config['reward_scale'] + 1e-5)

        grad_q = jax.grad(reward_fn)(X_final_clean)
        grad_q = jnp.nan_to_num(grad_q, nan=0.0, posinf=0.0, neginf=0.0)    

        adjoint = -jnp.nan_to_num(grad_q) # Removing 'q_grad_clip' to match paper strictly
        
        avg_reward = reward_fn(X_final_clean) / observations.shape[0]

        def scan_backward(adjoint, args):
            i, x_curr = args # x_curr is now correctly aligned (starts at X_N)
            t_float = (i + 1) / n_steps # Time corresponds to x_curr (t+h)
            
            # Note: t_safe logic in code used t_float + dt. 
            # If t_float is now (i+1)/N, it represents the time of 'adjoint'.
            # The dynamics are calculated at t_float.
            
            def drift_fn(a):
                return self.get_base_drift(observations, a, t_float)
            
            # Calculate dynamics at (adjoint, x_curr, t_float)
            v_base, vjp_fn = jax.vjp(drift_fn, x_curr)
            
            term1_grads = vjp_fn(adjoint)[0]
            term2_grads = - (1.0 / t_float) * adjoint
            
            # Eq 216: d(adj)/dt = -(2(grad v)^T adj - adj/t)
            # vjp_total represents NEGATIVE drift (for backward step)
            # Removing 'vjp_clip' to match paper strictly
            vjp_total = 2 * term1_grads + term2_grads
            
            # Step from t+h to t
            adjoint_next = adjoint + vjp_total * dt
            
            # --- Target Calculation ---
            # Use adjoint_next (approx a_t) and time t = i/n_steps
            t_target = i / n_steps
            sigma = jnp.sqrt(2 * (1 - t_target + dt/4) / (t_target + dt/4))
            
            # Re-fetch v_base at time t (since x_curr was at t+h)
            # This requires X at time t. 
            # Ideally, scan should carry (X_{t+h}, X_t).
            # APPROXIMATION FIX:
            # Since we calculate target for the loss at time t, 
            # we need v_base(X_t) and adjoint(X_t).
            # The scan output 'targets' matches indices 0..N-1.
            # We cannot easily re-calculate v_base(X_t) here without passing X_t in args.
            
            # BETTER APPROACH:
            # We return adjoint_next. We compute the final target in the loss loop 
            # or map it post-scan. 
            # HOWEVER, to keep structure:
            # We will return the raw adjoint_next and compute the full target later,
            # OR assume v_base doesn't change drastically between t and t+h (Euler approx).
            # To be STRICT: The target_v must be calculated outside or pass X_t in.
            
            return adjoint_next, adjoint_next # Return adjoint for external computation

        # 1. FIX INDEXING: Input starts at X_N (traj[-1]) down to X_1
        indices = jnp.arange(n_steps)[::-1] # N-1 ... 0
        traj_inputs = traj[1:][::-1]        # X_N ... X_1
        
        _, adjoint_seq = jax.lax.scan(scan_backward, adjoint, (indices, traj_inputs))
        
        # adjoint_seq is ordered N-1 ... 0. Reverse to 0 ... N-1
        adjoint_seq = adjoint_seq[::-1]
        
        # Now compute Targets using correct alignment:
        # X_t (traj[:-1]), Time t, and Adjoint_t (adjoint_seq)
        x_t_seq = traj[:-1]
        times = jnp.linspace(0, 1.0 - dt, n_steps)
        
        # Vectorized target computation to ensure exact t matching
        def compute_v_target(x, adj, t):
            sigma = jnp.sqrt(2 * (1 - t + dt/4) / (t + dt/4))
            v_b = self.get_base_drift(observations, x, t)
            # Use the CORRECTED SIGN from previous turn:
            return v_b - (0.5 * sigma**2) * adj
            
        final_targets = jax.vmap(compute_v_target)(x_t_seq, adjoint_seq, times)

        return final_targets, avg_reward

    @jax.jit
    def update(self, batch):
        rng = self.rng
        batch_size = batch['actions'].shape[0]
        
        n_steps = self.config.get('ode_steps', 20) 
        dt = 1.0 / n_steps
        
        rng, sample_rng, flow_rng, sub_rng = jax.random.split(rng, 4)
        
        # Forward
        traj, _ = self.forward_sde(flow_rng, batch['observations'], n_steps, dt)
        
        # Backward
        # NOTE: 'target_traj' contains the pre-calculated v_base + correction
        target_traj, avg_rew = self.compute_targets(traj, batch['observations'], n_steps, dt)
        
        # Subsampling
        k_last = 9
        m_random = 9
        last_indices = jnp.arange(n_steps - k_last, n_steps)
        rand_indices = jax.random.randint(sub_rng, (m_random,), 1, n_steps - k_last)
        active_indices = jnp.sort(jnp.concatenate([last_indices, rand_indices]))
        
        # Slice the data
        active_x_t = traj[active_indices]             
        active_targets = target_traj[active_indices]   # <--- This is v_target, not adjoint
        active_times = active_indices / n_steps       
        
        def loss_fn(params):
            def step_loss(x_t, v_target_t, t_val):
                t_batch = jnp.full((batch_size, 1), t_val)
                
                # 1. Student Prediction
                v_student = self.network.select('student_policy')(
                    batch['observations'], x_t, t_batch, params=params
                )
                
                # 2. Weighting
                sigma_t = jnp.sqrt(2 * (1 - t_val + dt/4) / (t_val + dt/4))
                weight = 4.0 / (sigma_t**2 + 1e-5)
                
                # 3. Loss (MSE against the pre-computed target)
                # We DO NOT re-calculate v_base or add adjoint here.
                sq_err = jnp.sum((v_student - v_target_t)**2, axis=-1)
                
                weighted_err = weight * sq_err
                return jnp.mean(jnp.clip(weighted_err, a_max=self.config['LCT']))

            losses = jax.vmap(step_loss)(active_x_t, active_targets, active_times)
            return jnp.mean(losses)

        grad_fn = jax.value_and_grad(loss_fn)
        loss_val, grads = grad_fn(self.network.params)
        new_network = self.network.apply_gradients(grads=grads)
        
        info = {"loss": loss_val, "avg_reward": avg_rew}
        return self.replace(network=new_network, rng=rng), info

def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='adjoint_matching',
        lr=3e-4,
        batch_size=256,
        actor_hidden_dims=(512, 512, 512, 512),
        actor_layer_norm=False,
        
        # Placeholders
        ode_steps=ml_collections.config_dict.placeholder(int),      
        uncertainty_beta=ml_collections.config_dict.placeholder(float), 
        reward_scale=ml_collections.config_dict.placeholder(float),
        LCT=ml_collections.config_dict.placeholder(float),
        action_dim=ml_collections.config_dict.placeholder(int),
        q_grad_clip=ml_collections.config_dict.placeholder(float),
        vjp_clip=ml_collections.config_dict.placeholder(float),
    ))
    return config