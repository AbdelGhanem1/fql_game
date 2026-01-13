import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
import optax
import flax
from flax.training import train_state
from typing import Any

# Import existing utils from the repo
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
from agents.iql import IQLAgent  # Using the REPO'S IQL Agent
from agents.am import AdjointMatchingAgent # Ensure agents/am.py is saved!

# --- 1. A Simple Flow BC Agent (No One-Step, No Critic) ---
class FlowBCAgent(flax.struct.PyTreeNode):
    rng: Any  # [FIXED] Added type hint
    network: Any
    config: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        
        # We only need the Flow Actor
        network_def = ModuleDict({
            'actor_bc_flow': ActorVectorField(
                hidden_dims=config['hidden_dims'],
                action_dim=ex_actions.shape[-1],
                layer_norm=False
            )
        })
        
        # Init
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
            
            # Flow Matching Logic
            rng_loc = jax.random.fold_in(rng, 0) # Simple rng handling for jit
            rng_x, rng_t = jax.random.split(rng_loc)
            
            x_0 = jax.random.normal(rng_x, actions.shape)
            x_1 = actions
            t = jax.random.uniform(rng_t, (batch_size, 1))
            
            x_t = (1 - t) * x_0 + t * x_1
            vel_target = x_1 - x_0
            
            # Predict
            pred = self.network.apply(
                {'params': params},
                observations=obs,
                actions=x_t,
                times=t,
                method=lambda m: m['actor_bc_flow'](obs, x_t, t)
            )
            return jnp.mean((pred - vel_target) ** 2)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        info['loss'] = info['loss'] # Keep consistent naming
        return self.replace(network=new_network, rng=new_rng), info

# --- 2. Data & Viz ---
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
             # Manual select for our custom lightweight agent
             v = agent.network.apply(
                {'params': agent.network.params},
                observations=obs, actions=x, times=t,
                method=lambda m: m['actor_bc_flow'](obs, x, t)
            )
        elif hasattr(agent, 'get_ode_drift'):
             # AM Agent
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

# --- 3. Main Loop ---
def main():
    seed = 42
    dummy_obs = jnp.zeros((1, 2))
    dummy_act = jnp.zeros((1, 2))
    dataset = get_toy_dataset()

    # --- A. Train IQL Critic (Reward Model) ---
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
        'const_std': True, # [FIX] Added missing config key
        'encoder': None
    })
    
    iql_agent = IQLAgent.create(seed, dummy_obs, dummy_act, iql_config)
    
    for i in range(2001):
        idxs = np.random.randint(0, dataset['observations'].shape[0], 256)
        batch = {k: v[idxs] for k, v in dataset.items()}
        # IQL update trains both Critic and Value
        iql_agent, info = iql_agent.update(batch)
        
        if i % 1000 == 0:
            print(f"Step {i} | V Loss: {info['value_loss']:.4f} | Q Loss: {info['critic_loss']:.4f}")

    # --- B. Train Base Flow Model (BC) ---
    print("\nPhase 2: Training Base Flow Model (BC)...")
    bc_config = ml_collections.ConfigDict({
        'lr': 1e-3,
        'hidden_dims': (64, 64)
    })
    
    bc_agent = FlowBCAgent.create(seed, dummy_obs, dummy_act, bc_config)
    
    for i in range(2001):
        idxs = np.random.randint(0, dataset['observations'].shape[0], 256)
        batch = {k: v[idxs] for k, v in dataset.items()}
        bc_agent, info = bc_agent.update(batch)
        
        if i % 1000 == 0:
            print(f"Step {i} | Flow BC Loss: {info['loss']:.4f}")
            
    plot_results(bc_agent, "Base BC Flow Policy")

    # --- C. Adjoint Matching ---
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

    # Initialize AM Agent
    # We pass 'bc_agent' as base, and 'iql_agent' as critic
    am_agent = AdjointMatchingAgent.create(
        seed=seed,
        ex_observations=dummy_obs,
        ex_actions=dummy_act,
        config=am_config,
        base_agent=bc_agent, 
        critic_agent=iql_agent
    )

    for i in range(1001):
        idxs = np.random.randint(0, dataset['observations'].shape[0], 256)
        batch = {k: v[idxs] for k, v in dataset.items()}
        am_agent, info = am_agent.update(batch)
        
        if i % 200 == 0:
            print(f"Step {i} | AM Loss: {info['loss']:.4f} | Avg Reward: {info['avg_reward']:.2f}")

    plot_results(am_agent, "Adjoint Matching Policy (Finetuned)")

if __name__ == "__main__":
    main()