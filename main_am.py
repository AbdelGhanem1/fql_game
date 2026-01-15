import os
import pickle
import random
import numpy as np
import jax
import jax.numpy as jnp
import ml_collections
from absl import app, flags
from ml_collections import config_flags
import tqdm
import flax 
import wandb

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
flags.DEFINE_integer('am_steps', 10000, 'Number of AM finetuning steps.')
flags.DEFINE_integer('base_train_steps', 100000, 'Steps for base training.')

# Evaluation Intervals
flags.DEFINE_integer('eval_interval', 10000, 'Interval for Base Training evaluation.')
flags.DEFINE_integer('am_eval_interval', 500, 'Interval for AM Finetuning evaluation.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_float('eval_temperature', 0.0, 'Temperature for evaluation.')

# AM Hyperparameters
flags.DEFINE_float('reward_scale', 1.0, 'Scale of the Q-guidance signal.')
flags.DEFINE_float('q_grad_clip', 10.0, 'Clip value for the gradient of the Q-function.')
flags.DEFINE_float('LCT', 10.0, 'Large Deviation Truncation threshold for loss.')
flags.DEFINE_float('vjp_clip', 10.0, 'Clip value for the vector-Jacobian product in adjoint.')
flags.DEFINE_integer('ode_steps', 20, 'Number of ODE steps for AM training and inference.')

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

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    # JAX seeding is handled by passing the key, but we set python/numpy for dataloaders

def main(_):
    # Ensure full reproducibility
    set_global_seed(FLAGS.seed)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    # Init WandB (expecting env vars for grouping)
    wandb.init(config=FLAGS.config.to_dict())
    
    # --- 1. Init Environment ---
    # make_env_and_datasets uses np.random, so global seed helps here
    env, eval_env, train_dataset_dict, _ = make_env_and_datasets(FLAGS.env_name, frame_stack=None)
    train_dataset = Dataset.create(**train_dataset_dict)
    example_batch = train_dataset.sample(1)
    


    # --- 2. Base Training (or Load) ---
    flow_agent = None
    critic_agent = None
    
    # Define the specific filename for THIS seed
    seed_base_flow_path = os.path.join(FLAGS.save_dir, f'base_flow_{FLAGS.env_name}_seed{FLAGS.seed}.pkl')
    seed_base_critic_path = os.path.join(FLAGS.save_dir, f'base_critic_{FLAGS.env_name}_seed{FLAGS.seed}.pkl')

    # LOGIC: If explicit paths are provided, use them. 
    # Otherwise, check if we already trained this seed before.
    if FLAGS.pretrained_flow_path != '':
        # Case A: Paths provided by launcher (e.g., for Scale 1.0 run)
        load_flow = FLAGS.pretrained_flow_path
        load_critic = FLAGS.pretrained_critic_path
        print(f"[Seed {FLAGS.seed}] Loading provided Base Agents:\n  - {load_flow}")
    elif os.path.exists(seed_base_flow_path):
        # Case B: No paths provided, BUT we found a saved file for this seed (Auto-Reuse)
        load_flow = seed_base_flow_path
        load_critic = seed_base_critic_path
        print(f"[Seed {FLAGS.seed}] Found existing Base Agents for this seed! Skipping Base Training.\n  - {load_flow}")
    else:
        # Case C: No paths, no file. Train from Scratch.
        load_flow = None
        load_critic = None

    if load_flow:
        # LOAD
        with open(load_flow, 'rb') as f:
            flow_state = pickle.load(f)
            flow_agent = FlowBCAgent.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], FLAGS.config.flow)
            flow_agent = flax.serialization.from_state_dict(flow_agent, flow_state)
        with open(load_critic, 'rb') as f:
            critic_state = pickle.load(f)
            critic_agent = IQLAgent.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], FLAGS.config.iql)
            critic_agent = flax.serialization.from_state_dict(critic_agent, critic_state)
    else:
        # TRAIN
        print(f"[Seed {FLAGS.seed}] No existing model found. Starting Base Training (100k steps)...")
        flow_agent, critic_agent = train_base_models(
            env_name=FLAGS.env_name,
            seed=FLAGS.seed,
            config=FLAGS.config,
            save_dir=FLAGS.save_dir,
            max_steps=FLAGS.base_train_steps,
            eval_interval=FLAGS.eval_interval,
            eval_episodes=FLAGS.eval_episodes,
            eval_temperature=FLAGS.eval_temperature 
        )
        
        # [CRITICAL] Save the specific seed copy immediately after training
        print(f"[Seed {FLAGS.seed}] Saving persistent base copy...")
        with open(seed_base_flow_path, 'wb') as f:
            pickle.dump(flax.serialization.to_state_dict(flow_agent), f)
        with open(seed_base_critic_path, 'wb') as f:
            pickle.dump(flax.serialization.to_state_dict(critic_agent), f)

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
    # Use deterministic sampling for baseline
    base_actions = flow_agent.sample_actions(monitor_obs, seed=jax.random.PRNGKey(0), temperature=0.0)
    q1_base, q2_base = critic_agent.network.select('target_critic')(monitor_obs, base_actions)
    mean_base_q = jnp.mean(jnp.minimum(q1_base, q2_base))
    
    wandb.log({"monitor/baseline_q": mean_base_q})

    # --- 5. Finetuning Loop ---
    print(f"[Seed {FLAGS.seed}] Starting AM Finetuning (Scale: {FLAGS.reward_scale}, 10k steps)...")
    
    # Progress bar
    pbar = tqdm.tqdm(range(1, FLAGS.am_steps + 1), smoothing=0.1, desc=f"AM-Scale-{FLAGS.reward_scale}-Seed-{FLAGS.seed}", mininterval=5.0, ncols=100)

    for i in pbar:
        batch = train_dataset.sample(FLAGS.config.am.batch_size)
        am_agent, info = am_agent.update(batch)
        
        # Monitor every 500 steps (matches eval interval)
        if i % 500 == 0:
            current_actions = am_agent.sample_actions(monitor_obs, seed=jax.random.PRNGKey(0), temperature=0.0)
            q1_curr, q2_curr = critic_agent.network.select('target_critic')(monitor_obs, current_actions)
            mean_curr_q = jnp.mean(jnp.minimum(q1_curr, q2_curr))
            
            q_improvement = mean_curr_q - mean_base_q
            action_drift = jnp.mean(jnp.linalg.norm(current_actions - base_actions, axis=-1))
            
            wandb.log({
                "am/loss": info['loss'],
                "monitor/delta_q": q_improvement,
                "monitor/action_drift": action_drift,
                "am_step": i
            })
            
        # Evaluation Logic (Every 500 steps)
        if i % FLAGS.am_eval_interval == 0:
            metrics, _, _ = evaluate(am_agent, eval_env, num_eval_episodes=FLAGS.eval_episodes, eval_temperature=FLAGS.eval_temperature)
            eval_score = metrics.get('episode.return', metrics.get('evaluation/return', -1000))
            
            try: normalized_score = env.unwrapped.get_normalized_score(eval_score) * 100.0
            except: normalized_score = eval_score
            
            wandb.log({
                "am/raw_score": eval_score,
                "am/norm_score": normalized_score,
                "am_step": i
            })
            
            pbar.set_description(f"AM (Drift:{action_drift:.2f}|Score:{normalized_score:.1f})")

if __name__ == '__main__':
    app.run(main)