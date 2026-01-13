# Save this as test_integration.py in the root of fql-master
import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
import ml_collections

# Import agents from the repo
from agents.fql import FQLAgent
from agents.am import AdjointMatchingAgent # Ensure you saved agents/am.py

# --- 1. Toy Data Generation (Wrapper for FQL) ---
def get_toy_dataset(n=4096):
    s_list, a_list, r_list, ns_list, d_list, m_list = [], [], [], [], [], []
    for _ in range(n):
        s = np.zeros(2)
        # Goal is [2,2]
        vec_goal = np.array([2,2]) - s
        noise = np.random.randn(2) * 0.2
        # Create suboptimal "detour" data for the Base Model to learn
        detour = np.array([-1, 1]) * np.random.uniform(1.0, 1.5)
        if np.random.rand() > 0.5: detour *= -1
        a = vec_goal + detour + noise
        a = a / np.linalg.norm(a) # Normalize actions to [-1, 1] roughly
        
        ns = s + a
        # Reward is negative distance
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

# --- 2. Visualization Helpers ---
def plot_results(agent, title):
    # Sample flow
    rng = jax.random.PRNGKey(0)
    # FQL/AM agents usually have a sample_actions method or we use the flow logic directly
    # Here we manually run the Euler loop using the network
    
    batch_size = 500
    action_dim = 2
    obs = jnp.zeros((batch_size, 2))
    
    # Generate Noise
    rng, x_rng = jax.random.split(rng)
    x = jax.random.normal(x_rng, (batch_size, action_dim))
    
    # Euler Integration (10 steps)
    dt = 1/10
    for i in range(10):
        t = jnp.full((batch_size, 1), i/10)
        # We need to handle selector depending on if it's FQL or AM agent
        if isinstance(agent, FQLAgent):
            v = agent.network.select('actor_bc_flow')(obs, x, t)
        else:
            # AM Agent
            v = agent.get_ode_drift(agent.network.params, 'modules_student_policy', obs, x, i/10)
        x = x + v * dt

    x = np.array(x)
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:,0], x[:,1], alpha=0.5, s=10)
    plt.xlim(-1, 3); plt.ylim(-1, 3)
    plt.arrow(0, 0, 2, 2, head_width=0.1, color='red') # Goal vector
    plt.title(title)
    plt.grid()
    plt.show()

# --- 3. Main Test Loop ---
def main():
    seed = 42
    rng = jax.random.PRNGKey(seed)
    
    # Config matching FQL defaults but simpler
    config = ml_collections.ConfigDict({
        'agent_name': 'fql',
        'ob_dims': [2],
        'action_dim': 2,
        'lr': 1e-3,
        'batch_size': 256,
        'actor_hidden_dims': (64, 64), # Small for toy
        'value_hidden_dims': (64, 64),
        'layer_norm': False,
        'actor_layer_norm': False,
        'discount': 0.99,
        'tau': 0.005,
        'q_agg': 'min',
        'alpha': 1.0, # BC weight
        'flow_steps': 10,
        'normalize_q_loss': False,
        'decay_steps': 10000 # Dummy
    })

    # A. Init Base Agent (FQL)
    print("Initializing FQL Agent (Base & Reward Model)...")
    # Note: FQLAgent.create expects ex_observations and ex_actions to infer shapes
    dummy_obs = jnp.zeros((1, 2))
    dummy_act = jnp.zeros((1, 2))
    
    base_agent = FQLAgent.create(seed, dummy_obs, dummy_act, config)
    
    # Get Data
    dataset = get_toy_dataset()
    
    # B. Train Base Agent
    print("\nPhase 1: Training Base FQL (BC + Critic)...")
    for i in range(2000): # Short training
        # Simple Batch Sampling
        idxs = np.random.randint(0, dataset['observations'].shape[0], 256)
        batch = {k: v[idxs] for k, v in dataset.items()}
        
        base_agent, info = base_agent.update(batch)
        
        if i % 500 == 0:
            print(f"Step {i} | Critic Loss: {info['critic_loss']:.4f} | BC Loss: {info['actor_loss']:.4f}")

    print(">> Base Training Done.")
    plot_results(base_agent, "Base FQL Policy (Should be scattered/detoured)")

    # C. Init Adjoint Matching Agent
    print("\nPhase 2: Initializing Adjoint Matching Agent...")
    
    # AM Config
    am_config = ml_collections.ConfigDict({
        'agent_name': 'adjoint_matching',
        'lr': 1e-4,
        'batch_size': 256,
        'actor_hidden_dims': (64, 64),
        'actor_layer_norm': False,
        'am_steps': 10, # Match flow steps
        'reward_scale': 5.0, # Crucial hyperparam
        'LCT': 100.0,
        'q_grad_clip': 10.0,
        'vjp_clip': 10.0,
        'action_dim': 2
    })
    
    # This requires agents/am.py to be correct
    am_agent = AdjointMatchingAgent.create(
        seed=seed,
        ex_observations=dummy_obs,
        ex_actions=dummy_act,
        config=am_config,
        base_agent=base_agent,   # Source of BC weights
        critic_agent=base_agent  # Source of Reward (Q)
    )

    # D. Train Adjoint Matching
    print("\nPhase 3: Fine-tuning with Adjoint Matching...")
    for i in range(1000):
        idxs = np.random.randint(0, dataset['observations'].shape[0], 256)
        batch = {k: v[idxs] for k, v in dataset.items()}
        
        am_agent, info = am_agent.update(batch)
        
        if i % 100 == 0:
            print(f"Step {i} | AM Loss: {info['loss']:.4f} | Avg Reward: {info['avg_reward']:.2f}")

    print(">> AM Training Done.")
    plot_results(am_agent, "Adjoint Matching Policy (Should cluster at Goal)")

if __name__ == "__main__":
    main()