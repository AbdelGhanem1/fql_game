import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import ml_collections
from absl import app, flags
from ml_collections import config_flags
import tqdm
import flax 
import wandb # <--- ADDED

from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.evaluation import evaluate
from agents.flow_bc import FlowBCAgent
from agents.iql import IQLAgent
from agents.am import AdjointMatchingAgent, get_config as get_am_config
from train_base import train_base_models

FLAGS = flags.FLAGS

# Environment & Paths
flags.DEFINE_string('env_name', 'halfcheetah-medium-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './saved_models/', 'Directory to save checkpoints.')
flags.DEFINE_string('pretrained_flow_path', '', 'Path to pretrained FlowBC agent.')
flags.DEFINE_string('pretrained_critic_path', '', 'Path to pretrained IQL agent.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

# Training Steps
flags.DEFINE_integer('am_steps', 200000, 'Number of AM finetuning steps.')
flags.DEFINE_integer('base_train_steps', 1000000, 'Steps for base training (if needed).')

# Evaluation Settings
flags.DEFINE_integer('eval_interval', 50000, 'Interval for Base Training evaluation.')
flags.DEFINE_integer('am_eval_interval', 5000, 'Interval for AM Finetuning evaluation.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_float('eval_temperature', 1.0, 'Temperature for evaluation (0=deterministic, 1=stochastic).')

# AM Hyperparameters
flags.DEFINE_float('reward_scale', 1.0, 'Scale of the Q-guidance signal.')
flags.DEFINE_float('q_grad_clip', 10.0, 'Clip value for the gradient of the Q-function.')
flags.DEFINE_float('LCT', 10.0, 'Large Deviation Truncation threshold for loss.')
flags.DEFINE_float('vjp_clip', 10.0, 'Clip value for the vector-Jacobian product in adjoint.')
flags.DEFINE_integer('ode_steps', 40, 'Number of ODE steps for AM training and inference.')

def get_full_config():
    flow = ml_collections.ConfigDict({
        'agent_name': 'flow_bc', 'lr': 3e-4, 'batch_size': 256,
        'actor_hidden_dims': (512, 512, 512, 512), 'actor_layer_norm': False,
        'flow_steps': 10, 'encoder': None,
        'action_dim': ml_collections.config_dict.placeholder(int),
    })
    iql = ml_collections.ConfigDict({
        'agent_name': 'iql', 'lr': 3e-4, 'batch_size': 256,
        'actor_hidden_dims': (256, 256), 'value_hidden_dims': (256, 256),
        'layer_norm': True, 'actor_layer_norm': False,
        'discount': 0.99, 'tau': 0.005, 'expectile': 0.7, 
        'actor_loss': 'awr', 'alpha': 10.0, 'const_std': True, 'encoder': None,
    })
    am = get_am_config()
    am.actor_hidden_dims = (512, 512, 512, 512)
    return ml_collections.ConfigDict({'flow': flow, 'iql': iql, 'am': am})

config_flags.DEFINE_config_dict('config', get_full_config(), 'Full configuration.')

def main(_):
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    # --- 0. INIT WANDB ---
    # We expect WANDB_PROJECT and WANDB_NAME to be set via env vars in the bash command
    wandb.init(config=FLAGS.config.to_dict())
    
    # --- 1. Init Environment ---
    print(f"Initializing {FLAGS.env_name}...")
    env, eval_env, train_dataset_dict, _ = make_env_and_datasets(FLAGS.env_name, frame_stack=None)
    train_dataset = Dataset.create(**train_dataset_dict)
    example_batch = train_dataset.sample(1)
    
    # --- 2. Load Pretrained Agents ---
    flow_agent = None
    critic_agent = None
    
    if FLAGS.pretrained_flow_path and FLAGS.pretrained_critic_path:
        print(f"Loading Agents...")
        flow_agent = FlowBCAgent.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], FLAGS.config.flow)
        critic_agent = IQLAgent.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], FLAGS.config.iql)
        
        with open(FLAGS.pretrained_flow_path, 'rb') as f:
            flow_state = pickle.load(f)
            flow_agent = flax.serialization.from_state_dict(flow_agent, flow_state)
        with open(FLAGS.pretrained_critic_path, 'rb') as f:
            critic_state = pickle.load(f)
            critic_agent = flax.serialization.from_state_dict(critic_agent, critic_state)
    else:
        raise ValueError("For ablation, please provide pretrained paths!")

    # --- 3. Initialize Adjoint Matching ---
    FLAGS.config.am.action_dim = example_batch['actions'].shape[-1]
    FLAGS.config.am.reward_scale = FLAGS.reward_scale
    FLAGS.config.am.LCT = FLAGS.LCT
    FLAGS.config.am.q_grad_clip = FLAGS.q_grad_clip
    FLAGS.config.am.vjp_clip = FLAGS.vjp_clip
    FLAGS.config.am.am_steps = FLAGS.ode_steps

    am_agent = AdjointMatchingAgent.create(
        seed=FLAGS.seed,
        ex_observations=example_batch['observations'],
        ex_actions=example_batch['actions'],
        config=FLAGS.config.am,
        base_agent=flow_agent,
        critic_agent=critic_agent
    )

    # --- 4. Setup Fixed Batch Monitor ---
    monitor_batch = train_dataset.sample(32)
    monitor_obs = monitor_batch['observations']
    base_actions = flow_agent.sample_actions(monitor_obs, seed=jax.random.PRNGKey(0), temperature=0.0)
    q1_base, q2_base = critic_agent.network.select('target_critic')(monitor_obs, base_actions)
    mean_base_q = jnp.mean(jnp.minimum(q1_base, q2_base))
    
    wandb.log({"monitor/baseline_q": mean_base_q})

    # --- 5. Finetuning Loop ---
    print(f"Starting AM Finetuning (Scale: {FLAGS.reward_scale})...")
    best_am_score = -float('inf')
    
    # Progress bar only!
    pbar = tqdm.tqdm(range(1, FLAGS.am_steps + 1), smoothing=0.1, desc=f"AM (Scale {FLAGS.reward_scale})", mininterval=5.0, ncols=100)

    for i in pbar:
        batch = train_dataset.sample(FLAGS.config.am.batch_size)
        am_agent, info = am_agent.update(batch)
        
        # --- Monitor Logic (Every 1000 steps) ---
        if i % 1000 == 0:
            current_actions = am_agent.sample_actions(monitor_obs, seed=jax.random.PRNGKey(0), temperature=0.0)
            q1_curr, q2_curr = critic_agent.network.select('target_critic')(monitor_obs, current_actions)
            mean_curr_q = jnp.mean(jnp.minimum(q1_curr, q2_curr))
            
            q_improvement = mean_curr_q - mean_base_q
            action_drift = jnp.mean(jnp.linalg.norm(current_actions - base_actions, axis=-1))
            
            # Log to WandB
            wandb.log({
                "train/loss": info['loss'],
                "train/proxy_reward": info['avg_reward'],
                "monitor/delta_q": q_improvement,
                "monitor/action_drift": action_drift,
                "monitor/mean_q": mean_curr_q,
                "step": i
            })

            # Update Pbar (Keep it minimal)
            pbar.set_description(f"AM (L:{info['loss']:.2f}|dQ:{q_improvement:+.2f}|Dr:{action_drift:.2f})")
            
        # --- Evaluation Logic ---
        if i % FLAGS.am_eval_interval == 0:
            metrics, _, _ = evaluate(am_agent, eval_env, num_eval_episodes=FLAGS.eval_episodes, eval_temperature=FLAGS.eval_temperature)
            eval_score = metrics.get('episode.return', metrics.get('evaluation/return', -1000))
            
            try: normalized_score = env.unwrapped.get_normalized_score(eval_score) * 100.0
            except: normalized_score = eval_score
            
            # Log Eval to WandB
            wandb.log({
                "eval/raw_score": eval_score,
                "eval/norm_score": normalized_score,
                "step": i
            })
            
            # Minimal Print (Optional, can rely purely on WandB)
            # tqdm.tqdm.write(f"Step {i}: Score {normalized_score:.2f}")

            if eval_score > best_am_score:
                best_am_score = eval_score
                # Only save if explicitly needed, usually ablation doesn't need heavy saving
                # or save with scale in name
                save_path = os.path.join(FLAGS.save_dir, f'am_pen_{FLAGS.reward_scale}_best.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(flax.serialization.to_state_dict(am_agent), f)

if __name__ == '__main__':
    app.run(main)