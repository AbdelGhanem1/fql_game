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

        # --- Verify Output ---
        print("--- Verifying Output Similarity ---")
        dummy_obs = ex_observations[:1]
        dummy_act = ex_actions[:1]
        dummy_t = jnp.zeros((1, 1))

        if hasattr(base_agent.network, 'select'):
             v_base = base_agent.network.select('actor_bc_flow')(dummy_obs, dummy_act, dummy_t)
        else:
             v_base = base_agent.network.apply(
                {'params': base_agent.network.params},
                dummy_obs, dummy_act, dummy_t,
                method=lambda m: m.actor_bc_flow(dummy_obs, dummy_act, dummy_t) 
            )

        v_student = network.select('student_policy')(
            dummy_obs, dummy_act, dummy_t, params=network.params
        )

        diff = jnp.mean((v_base - v_student) ** 2)
        print(f"   MSE Diff: {float(diff):.6f}")
        
        if diff > 1e-5:
            print("❌ CRITICAL FAIL: Student outputs do not match Base Agent!")
        else:
            print("✅ SUCCESS: Student is a perfect clone of Base Agent.")
        print("=======================================\n")

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
        
        steps = self.config.get('am_steps', 10) 
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
        
        v_base_final = self.get_base_drift(observations, X_pre_final, t_pre_final)
        X_final_clean = X_pre_final + v_base_final * dt
        
        def reward_fn(a):
            # Clip actions before IQL critic
            a_clipped = jnp.clip(a, -1.0, 1.0)
            
            # 1. Get all ensemble Q-values
            qs = self.critic_agent.network.select('target_critic')(observations, actions=a_clipped)
            
            if isinstance(qs, (list, tuple)):
                qs_stack = jnp.stack(qs, axis=0)
            else:
                qs_stack = qs[None, ...] 

            # 2. Compute Stats
            q_mean = jnp.mean(qs_stack, axis=0)
            
            # [CRITICAL FIX] Stabilize Std Gradient
            # If critics agree perfectly, variance is 0. 
            # The gradient of sqrt(0) is infinite. We add 1e-6 to prevent this.
            q_var = jnp.var(qs_stack, axis=0)
            q_std = jnp.sqrt(q_var + 1e-6)
            
            # 3. Apply Uncertainty Penalty (LCB)
            beta = self.config.get('uncertainty_beta', 2.0)
            robust_q = q_mean - beta * q_std
            
            return jnp.sum(robust_q) * self.config['reward_scale']

        grad_q = jax.grad(reward_fn)(X_final_clean)
        
        # [Additional Safety] Explicitly sanitize gradients to prevent NaNs propagating
        grad_q = jnp.nan_to_num(grad_q, nan=0.0, posinf=0.0, neginf=0.0)
        
        adjoint = -jnp.clip(grad_q, -self.config['q_grad_clip'], self.config['q_grad_clip'])
        
        avg_reward = reward_fn(X_final_clean) / (self.config['reward_scale'] * observations.shape[0])

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
            
            sigma = jnp.sqrt(2 * (1 - t_float + dt) / (t_float + dt))
            target_v = v_base - (0.5 * sigma**2) * adjoint
            
            return adjoint_next, target_v

        indices = jnp.arange(n_steps)[::-1]
        traj_inputs = traj[:-1][::-1]
        _, targets = jax.lax.scan(scan_backward, adjoint, (indices, traj_inputs))
        
        return targets[::-1], avg_reward

    @functools.partial(jax.jit, static_argnames=('batch_size',))
    def update(self, batch):
        rng = self.rng
        batch_size = batch['actions'].shape[0]
        
        # --- 1. Setup & Constants ---
        n_steps = self.config['ode_steps'] # e.g., 40
        dt = 1.0 / n_steps
        scale = self.config['reward_scale']
        
        # Split RNGs
        rng, sample_rng, flow_rng, sub_rng = jax.random.split(rng, 4)
        
        # --- 2. Forward SDE (Generate Trajectories) ---
        # We must run this fully to get the trajectory
        traj, noise = self.forward_sde(batch, n_steps, flow_rng)
        # traj shape: (n_steps + 1, batch_size, action_dim)
        
        # --- 3. Define Reward Function with Robust Scaling ---
        # We divide by scale here so gradients are normalized
        def scaled_reward_fn(x):
            # Calculate Raw Q-values (LCB or Mean)
            qs = self.critic_agent.network.select('target_critic')(batch['observations'], x)
            if isinstance(qs, (list, tuple)):
                qs = jnp.stack(qs, axis=0) # (Ens, Batch)
            
            # LCB for robustness (OOD protection)
            q_mean = jnp.mean(qs, axis=0)
            q_var = jnp.var(qs, axis=0)
            beta = self.config.get('uncertainty_beta', 1.0)
            raw_reward = q_mean - beta * jnp.sqrt(q_var)
            
            # NORMALIZATION: Robust Scaling
            return jnp.sum(raw_reward) / scale

        # --- 4. Backward ODE (Compute Adjoints) ---
        # We must run this fully to propagate adjoints from t=1 to t=0
        adjoint_traj = self.compute_targets(traj, batch['observations'], dt, scaled_reward_fn)
        # adjoint_traj shape: (n_steps + 1, batch_size, action_dim)
        
        # --- 5. Efficient Subsampling (Speed & Stability Optimization) ---
        # Strategy: Train on last k steps (structure) + m random steps (exploration)
        # This skips the noisy t=0 steps and reduces compute.
        
        k_last = 10 # Train heavily on refinement steps
        m_random = 10 # Random samples from the rest
        
        # Indices for the end of trajectory (t near 1)
        # We use steps 0 to n_steps-1 for v_student calls
        last_indices = jnp.arange(n_steps - k_last, n_steps)
        
        # Random indices from the earlier part (excluding t=0 for stability)
        rand_indices = jax.random.randint(sub_rng, (m_random,), 1, n_steps - k_last)
        
        # Combine and sort
        active_indices = jnp.sort(jnp.concatenate([last_indices, rand_indices]))
        
        # SLICE: Extract only the data we need for the loss
        # We drop the last element of traj for input x_t (since we predict velocity at t)
        active_x_t = traj[active_indices]           # (n_active, B, D)
        active_adjoint = adjoint_traj[active_indices] # (n_active, B, D)
        active_times = active_indices / n_steps       # (n_active,)
        
        # --- 6. Loss Computation (The "Loop") ---
        def loss_fn(params):
            # Helper to compute loss for a single timestep t
            def step_loss(x_t, a_t, t_val):
                # Expand time for batch
                t_batch = jnp.full((batch_size, 1), t_val)
                
                # a. Base Model Drift (Frozen)
                v_base = self.base_network.select('actor_bc_flow')(batch['observations'], x_t, t_batch)
                
                # b. Student Model Drift (Trainable)
                v_student = self.network.select('student_policy')(
                    batch['observations'], x_t, t_batch, params=params
                )
                
                # c. Sigma with singularity offsets (Appendix H.1)
                # Note: t_val is discrete index/n_steps. 
                sigma_t = jnp.sqrt(2 * (1 - t_val + dt) / (t_val + dt))
                
                # d. Regression Target (Adjoint Matching)
                # v_target = v_base - (sigma^2 / 2) * adjoint
                target = v_base - (0.5 * sigma_t**2) * a_t
                
                # e. Weighting & Loss
                # Weight = 4 / sigma^2
                weight = 4.0 / (sigma_t**2 + 1e-5)
                sq_err = jnp.sum((v_student - target)**2, axis=-1)
                
                # f. LCT (Loss Clipping)
                # We expect LCT ~ 1.6 since we normalized rewards
                weighted_err = weight * sq_err
                return jnp.mean(jnp.clip(weighted_err, a_max=self.config['LCT']))

            # Vectorize over the ACTIVE time steps only
            losses = jax.vmap(step_loss)(active_x_t, active_adjoint, active_times)
            return jnp.mean(losses)

        # --- 7. Gradient Update ---
        grad_fn = jax.value_and_grad(loss_fn)
        loss_val, grads = grad_fn(self.network.params)
        new_network = self.network.apply_gradients(grads=grads)
        
        return self.replace(network=new_network, rng=rng), {"loss": loss_val}

def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='adjoint_matching',
        lr=1e-5,
        batch_size=256,
        actor_hidden_dims=(512, 512, 512, 512),
        actor_layer_norm=False,
        am_steps=ml_collections.config_dict.placeholder(int),
        reward_scale=ml_collections.config_dict.placeholder(float),
        LCT=ml_collections.config_dict.placeholder(float),
        q_grad_clip=ml_collections.config_dict.placeholder(float),
        vjp_clip=ml_collections.config_dict.placeholder(float),
        action_dim=ml_collections.config_dict.placeholder(int),
        uncertainty_beta=ml_collections.config_dict.placeholder(float), # NEW
    ))
    return config