import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
import optax
import flax
import functools
from flax.training import train_state
from typing import Any

# Import existing utils from the repo
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
from agents.iql import IQLAgent

# --- 1. Flow BC Agent (Fixed to return tuple (loss, info)) ---
class FlowBCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        
        network_def = ModuleDict({
            'actor_bc_flow': ActorVectorField(
                hidden_dims=config['hidden_dims'],
                action_dim=ex_actions.shape[-1],
                layer_norm=False
            )
        })
        
        ex_times = jnp.zeros((ex_actions.shape[0], 1))
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, 
                                          actor_bc_flow=(ex_observations, ex_actions, ex_times))['params']
        
        network = TrainState.create(network_def, network_params, tx=network_tx)
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        
        def loss_fn(params):
            obs = batch['observations']
            actions = batch['actions']
            batch_size = actions.shape[0]
            
            rng_loc = jax.random.fold_in(rng, 0)
            rng_x, rng_t = jax.random.split(rng_loc)
            
            x_0 = jax.random.normal(rng_x, actions.shape)
            x_1 = actions
            t = jax.random.uniform(rng_t, (batch_size, 1))
            
            x_t = (1 - t) * x_0 + t * x_1
            vel_target = x_1 - x_0
            
            pred = self.network.apply(
                {'params': params},
                observations=obs,
                actions=x_t,
                times=t,
                method=lambda m: m['actor_bc_flow'](obs, x_t, t)
            )
            loss = jnp.mean((pred - vel_target) ** 2)
            return loss, {'loss': loss} # [CRITICAL FIX] Return tuple

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

# --- 2. Adjoint Matching Agent (Included here to ensure correct Tuple return) ---
class AdjointMatchingAgent(flax.struct.PyTreeNode):
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

        # Logic to handle different base agents
        if hasattr(base_agent.network, 'params'):
            params = base_agent.network.params
            if 'modules_actor_bc_flow' in params:
                base_params = params['modules_actor_bc_flow']
            elif 'actor_bc_flow' in params: # Handle FlowBCAgent structure
                base_params = params['actor_bc_flow']
            else:
                 # Fallback for simple dict structure
                base_params = params
        else:
             raise ValueError("Unknown Base Agent Structure")

        network_tx = optax.adam(learning_rate=config['lr'])
        dummy_time = jnp.zeros((1, 1))
        
        init_params = network_def.init(init_rng, 
                                      observations=ex_observations[:1], 
                                      actions=ex_actions[:1], 
                                      times=dummy_time)['params']
        
        init_params['modules_student_policy'] = base_params
        network = TrainState.create(network_def, init_params, tx=network_tx)

        return cls(rng=rng, network=network, base_network=base_agent.network, critic_agent=critic_agent, config=flax.core.FrozenDict(**config))

    def get_ode_drift(self, params, module_name, observations, actions, t_scalar):
        batch_size = actions.shape[0]
        times = jnp.full((batch_size, 1), t_scalar)
        return self.network.apply(
            {'params': params},
            observations=observations, actions=actions, times=times,
            method=lambda module: module[module_name](observations, actions, times)
        )
    
    def get_base_drift(self, observations, actions, t_scalar):
        batch_size = actions.shape[0]
        times = jnp.full((batch_size, 1), t_scalar)
        # Handle FlowBCAgent structure vs FQLAgent structure
        if 'actor_bc_flow' in self.base_network.params:
             return self.base_network.apply(
                {'params': self.base_network.params},
                observations=observations, actions=actions, times=times,
                method=lambda m: m['actor_bc_flow'](observations, actions, times)
            )
        else:
            return self.base_network.select('actor_bc_flow')(observations, actions, times)

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
            
            v_stud = self.get_ode_drift(self.network.params, 'modules_student_policy', observations, a_t, t_float)
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
            q1, q2 = self.critic_agent.network.select('target_critic')(observations, actions=a)
            min_q = jnp.minimum(q1, q2)
            return jnp.sum(min_q) * self.config['reward_scale']

        grad_q = jax.grad(reward_fn)(X_final_clean)
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

    @jax.jit
    def update(self, batch):
        rng, step_rng = jax.random.split(self.rng)
        observations = batch['observations']
        n_steps = self.config['am_steps']
        dt = 1.0 / n_steps

        traj = self.forward_sde(step_rng, observations, n_steps, dt)
        targets, avg_rew = self.compute_targets(traj, observations, n_steps, dt)
        
        def loss_fn(params):
            def get_drift(i, x_t):
                return self.get_ode_drift(params, 'modules_student_policy', observations, x_t, i / n_steps)
            preds = jax.vmap(get_drift)(jnp.arange(n_steps), traj[:-1])
            sq_err = jnp.sum((preds - targets)**2, axis=-1)
            loss = jnp.mean(jnp.clip(sq_err, a_max=self.config['LCT']))
            return loss, {'loss': loss} # [CRITICAL FIX] Return tuple

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        info['avg_reward'] = avg_rew
        return self.replace(network=new_network, rng=rng), info

# --- 3. Dataset & Plotting ---
def get_toy_dataset(n=4096):
    s_list, a_list, r_list, ns_list, d_list, m_list = [], [], [], [], [], []
    for _ in range(n):
        s = np.zeros(2)
        vec_goal = np.array([2,2]) - s
        noise = np.random.randn(2) * 0.2
        detour = np.array([-1, 1]) * np.random.uniform(1.0, 1.5)
        if np.random.rand() > 0.5: detour *= -1
        a = vec_goal + detour + noise
        a = a / np.linalg.norm(a)
        ns = s + a
        r = -np.linalg.norm(ns - np.array([2,2]))
        s_list.append(s); a_list.append(a); r_list.append(r)
        ns_list.append(ns); d_list.append(0.0); m_list.append(1.0)
    return {
        'observations': np.array(s_list, dtype=np.float32),
        'actions': np.array(a_list, dtype=np.float32),
        'rewards': np.array(r_list, dtype=np.float32),
        'next_observations': np.array(ns_list, dtype=np.float32),
        'terminals': np.array(d_list, dtype=np.float32),
        'masks': np.array(m_list, dtype=np.float32)
    }

def plot_results(agent, title):
    rng = jax.random.PRNGKey(0)
    batch_size = 500
    obs = jnp.zeros((batch_size, 2))
    rng, x_rng = jax.random.split(rng)
    x = jax.random.normal(x_rng, (batch_size, 2))
    dt = 1/10
    for i in range(10):
        t = jnp.full((batch_size, 1), i/10)
        if isinstance(agent, FlowBCAgent):
             v = agent.network.apply(
                {'params': agent.network.params},
                observations=obs, actions=x, times=t,
                method=lambda m: m['actor_bc_flow'](obs, x, t)
            )
        elif hasattr(agent, 'get_ode_drift'):
             v = agent.get_ode_drift(agent.network.params, 'modules_student_policy', obs, x, i/10)
        x = x + v * dt
    x = np.array(x)
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:,0], x[:,1], alpha=0.5, s=10)
    plt.xlim(-1, 3); plt.ylim(-1, 3)
    plt.arrow(0, 0, 2, 2, head_width=0.1, color='red')
    plt.title(title)
    plt.grid()
    plt.show()

# --- 4. Main ---
def main():
    seed = 42
    dummy_obs = jnp.zeros((1, 2))
    dummy_act = jnp.zeros((1, 2))
    dataset = get_toy_dataset()

    print("Phase 1: Training IQL Critic...")
    iql_config = ml_collections.ConfigDict({
        'agent_name': 'iql',
        'lr': 3e-4,
        'batch_size': 256,
        'actor_hidden_dims': (64, 64),
        'value_hidden_dims': (64, 64),
        'layer_norm': False,
        'actor_layer_norm': False,
        'discount': 0.99,
        'tau': 0.005,
        'expectile': 0.7, 
        'actor_loss': 'awr', 
        'alpha': 10.0,
        'const_std': True, # Corrected Config
        'encoder': None
    })
    
    iql_agent = IQLAgent.create(seed, dummy_obs, dummy_act, iql_config)
    
    for i in range(2001):
        idxs = np.random.randint(0, dataset['observations'].shape[0], 256)
        batch = {k: v[idxs] for k, v in dataset.items()}
        iql_agent, info = iql_agent.update(batch)
        if i % 1000 == 0:
            # Corrected Keys
            print(f"Step {i} | V Loss: {info['value/value_loss']:.4f} | Q Loss: {info['critic/critic_loss']:.4f}")

    print("\nPhase 2: Training Base Flow Model (BC)...")
    bc_config = ml_collections.ConfigDict({'lr': 1e-3, 'hidden_dims': (64, 64)})
    bc_agent = FlowBCAgent.create(seed, dummy_obs, dummy_act, bc_config)
    
    for i in range(2001):
        idxs = np.random.randint(0, dataset['observations'].shape[0], 256)
        batch = {k: v[idxs] for k, v in dataset.items()}
        bc_agent, info = bc_agent.update(batch)
        if i % 1000 == 0:
            print(f"Step {i} | Flow BC Loss: {info['loss']:.4f}")
    plot_results(bc_agent, "Base BC Flow Policy")

    print("\nPhase 3: Adjoint Matching...")
    am_config = ml_collections.ConfigDict({
        'agent_name': 'adjoint_matching',
        'lr': 1e-4,
        'batch_size': 256,
        'actor_hidden_dims': (64, 64),
        'actor_layer_norm': False,
        'am_steps': 10,
        'reward_scale': 5.0,
        'LCT': 100.0,
        'q_grad_clip': 10.0,
        'vjp_clip': 10.0,
        'action_dim': 2
    })

    am_agent = AdjointMatchingAgent.create(seed, dummy_obs, dummy_act, am_config, bc_agent, iql_agent)

    for i in range(1001):
        idxs = np.random.randint(0, dataset['observations'].shape[0], 256)
        batch = {k: v[idxs] for k, v in dataset.items()}
        am_agent, info = am_agent.update(batch)
        if i % 200 == 0:
            print(f"Step {i} | AM Loss: {info['loss']:.4f} | Avg Reward: {info['avg_reward']:.2f}")

    plot_results(am_agent, "Adjoint Matching Policy (Finetuned)")

if __name__ == "__main__":
    main()