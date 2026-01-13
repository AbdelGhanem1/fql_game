import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
from agents.fql import FQLAgent
from agents.am import AdjointMatchingAgent 

# --- 1. Toy Data Generation ---
def get_toy_dataset(n=4096):
    s_list, a_list, r_list, ns_list, d_list, m_list = [], [], [], [], [], []
    for _ in range(n):
        s = np.zeros(2)
        # Goal is [2,2]
        vec_goal = np.array([2,2]) - s
        noise = np.random.randn(2) * 0.2
        # Detour data for Base Model
        detour = np.array([-1, 1]) * np.random.uniform(1.0, 1.5)
        if np.random.rand() > 0.5: detour *= -1
        a = vec_goal + detour + noise
        a = a / np.linalg.norm(a) # Normalize
        
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

# --- 2. Visualization ---
def plot_results(agent, title):
    rng = jax.random.PRNGKey(0)
    batch_size = 500
    obs = jnp.zeros((batch_size, 2))
    
    rng, x_rng = jax.random.split(rng)
    x = jax.random.normal(x_rng, (batch_size, 2))
    
    # Euler Integration
    dt = 1/10
    for i in range(10):
        t = jnp.full((batch_size, 1), i/10)
        if hasattr(agent, 'network') and isinstance(agent, FQLAgent):
            # FQL Agent uses 'actor_bc_flow'
            v = agent.network.select('actor_bc_flow')(obs, x, t)
        else:
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

# --- 3. Main Test Loop ---
def main():
    seed = 42
    
    # [CORRECTED] Full Config for FQLAgent
    config = ml_collections.ConfigDict({
        'agent_name': 'fql',
        'lr': 1e-3,
        'batch_size': 256,
        'actor_hidden_dims': (64, 64),
        'value_hidden_dims': (64, 64),
        'layer_norm': False,
        'actor_layer_norm': False,
        'discount': 0.99,
        'tau': 0.005,
        'q_agg': 'min',         # Missing in previous version
        'alpha': 1.0,
        'flow_steps': 10,
        'normalize_q_loss': False,
        'encoder': None         # Crucial Fix!
    })

    print("Initializing FQL Agent...")
    dummy_obs = jnp.zeros((1, 2))
    dummy_act = jnp.zeros((1, 2))
    
    # FQLAgent.create infers dims from dummy data
    base_agent = FQLAgent.create(seed, dummy_obs, dummy_act, config)
    
    dataset = get_toy_dataset()
    
    print("\nPhase 1: Training Base FQL (BC + Critic)...")
    # Using TQDM for progress bar
    for i in range(2001): 
        idxs = np.random.randint(0, dataset['observations'].shape[0], 256)
        batch = {k: v[idxs] for k, v in dataset.items()}
        
        base_agent, info = base_agent.update(batch)
        
        if i % 500 == 0:
            print(f"Step {i} | Critic Loss: {info['critic_loss']:.4f} | BC Loss: {info['actor_loss']:.4f}")

    print(">> Base Training Done.")
    plot_results(base_agent, "Base FQL Policy")

    # --- Adjoint Matching Integration ---
    print("\nPhase 2: Initializing Adjoint Matching Agent...")
    
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
    
    # Create AM Agent using the pre-trained FQL Agent
    am_agent = AdjointMatchingAgent.create(
        seed=seed,
        ex_observations=dummy_obs,
        ex_actions=dummy_act,
        config=am_config,
        base_agent=base_agent,
        critic_agent=base_agent
    )

    print("\nPhase 3: Fine-tuning with Adjoint Matching...")
    for i in range(1001):
        idxs = np.random.randint(0, dataset['observations'].shape[0], 256)
        batch = {k: v[idxs] for k, v in dataset.items()}
        
        am_agent, info = am_agent.update(batch)
        
        if i % 200 == 0:
            print(f"Step {i} | AM Loss: {info['loss']:.4f} | Avg Reward: {info['avg_reward']:.2f}")

    print(">> AM Training Done.")
    plot_results(am_agent, "Adjoint Matching Policy (Finetuned)")

if __name__ == "__main__":
    main()