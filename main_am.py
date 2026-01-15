import os
import pickle
import numpy as np
import jax
import ml_collections
from absl import app, flags
from ml_collections import config_flags
import tqdm
import flax 

from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.evaluation import evaluate
from agents.flow_bc import FlowBCAgent
from agents.iql import IQLAgent
from agents.am import AdjointMatchingAgent, get_config as get_am_config
from train_base import train_base_models

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-medium-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './saved_models/', 'Directory to save checkpoints.')
flags.DEFINE_string('pretrained_flow_path', '', 'Path to pretrained FlowBC agent.')
flags.DEFINE_string('pretrained_critic_path', '', 'Path to pretrained IQL agent.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('am_steps', 200000, 'Number of AM finetuning steps.')
flags.DEFINE_integer('base_train_steps', 1000000, 'Steps for base training (if needed).')
flags.DEFINE_integer('eval_interval', 10000, 'Evaluation interval.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_float('eval_temperature', 1.0, 'Temperature for evaluation (0=deterministic, 1=stochastic).')

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
    
    # --- 1. ALWAYS Init Environment First ---
    print(f"Initializing {FLAGS.env_name}...")
    env, eval_env, train_dataset_dict, _ = make_env_and_datasets(FLAGS.env_name, frame_stack=None)
    train_dataset = Dataset.create(**train_dataset_dict)
    example_batch = train_dataset.sample(1)
    
    # Setup Agents
    flow_agent = None
    critic_agent = None
    has_pretrained = (FLAGS.pretrained_flow_path != '') and (FLAGS.pretrained_critic_path != '')
    
    if has_pretrained:
        print(f"Loading Pretrained Agents from disk...")
        flow_agent = FlowBCAgent.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], FLAGS.config.flow)
        critic_agent = IQLAgent.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], FLAGS.config.iql)
        
        with open(FLAGS.pretrained_flow_path, 'rb') as f:
            flow_state = pickle.load(f)
            flow_agent = flax.serialization.from_state_dict(flow_agent, flow_state)
        with open(FLAGS.pretrained_critic_path, 'rb') as f:
            critic_state = pickle.load(f)
            critic_agent = flax.serialization.from_state_dict(critic_agent, critic_state)
        print("Agents successfully restored.")
    else:
        print("Pretrained paths not provided. Triggering Base Training...")
        # Pass flags to train_base
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

    # --- 2. Initialize Adjoint Matching ---
    print(f"Initializing Adjoint Matching Agent...")
    FLAGS.config.am.action_dim = example_batch['actions'].shape[-1]
    
    am_agent = AdjointMatchingAgent.create(
        seed=FLAGS.seed,
        ex_observations=example_batch['observations'],
        ex_actions=example_batch['actions'],
        config=FLAGS.config.am,
        base_agent=flow_agent,
        critic_agent=critic_agent
    )

    # --- 3. Finetuning Loop ---
    print("Starting Adjoint Matching Finetuning...")
    best_am_score = -float('inf')
    
    # Baseline Eval
    metrics, _, _ = evaluate(flow_agent, eval_env, num_eval_episodes=FLAGS.eval_episodes, eval_temperature=FLAGS.eval_temperature)
    base_raw = metrics.get('episode.return', metrics.get('evaluation/return', -1000))
    try: norm_base = env.unwrapped.get_normalized_score(base_raw) * 100.0
    except: norm_base = base_raw
    print(f"Baseline (BC) Normalized Score: {norm_base:.2f}")

    pbar = tqdm.tqdm(range(1, FLAGS.am_steps + 1), smoothing=0.1, desc="AM Finetuning", mininterval=5.0, ncols=100)

    for i in pbar:
        batch = train_dataset.sample(FLAGS.config.am.batch_size)
        am_agent, info = am_agent.update(batch)
        
        if i % 1000 == 0:
            pbar.set_description(f"AM Finetuning (Loss: {info['loss']:.4f} | Rew: {info['avg_reward']:.2f})")
            
        if i % 1000 == 0:
            print(f"\n--- Eval Triggered at Step {i} ---")
            print(f"    Train Metrics > AM Loss: {info['loss']:.4f} | Proxy Reward: {info['avg_reward']:.2f}")
            print("    Running Eval...")
            
            metrics, _, _ = evaluate(am_agent, eval_env, num_eval_episodes=FLAGS.eval_episodes, eval_temperature=FLAGS.eval_temperature)
            
            # [CRITICAL FIX] Correct key
            eval_score = metrics.get('episode.return', metrics.get('evaluation/return', -1000))
            
            try: normalized_score = env.unwrapped.get_normalized_score(eval_score) * 100.0
            except: normalized_score = eval_score
            
            print(f"    Eval Result   > Raw: {eval_score:.2f} | Norm: {normalized_score:.2f}")
            print(f"--------------------------------\n")
            
            if eval_score > best_am_score:
                best_am_score = eval_score
                with open(os.path.join(FLAGS.save_dir, f'am_finetuned_{FLAGS.env_name}.pkl'), 'wb') as f:
                    pickle.dump(flax.serialization.to_state_dict(am_agent), f)
                print(f"    Saved New Best Model")

if __name__ == '__main__':
    app.run(main)