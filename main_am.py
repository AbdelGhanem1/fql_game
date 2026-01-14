import os
import pickle
import numpy as np
import jax
import ml_collections
from absl import app, flags
from ml_collections import config_flags
import tqdm
import gym
import d4rl

from utils.datasets import Dataset
from agents.flow_bc import FlowBCAgent
from agents.iql import IQLAgent
from agents.am import AdjointMatchingAgent, get_config as get_am_config
from train_base import train_base_models, make_env, eval_policy # Import helpers

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-medium-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './saved_models/', 'Directory to save checkpoints.')
flags.DEFINE_string('pretrained_flow_path', '', 'Path to pretrained FlowBC agent. If empty, trains from scratch.')
flags.DEFINE_string('pretrained_critic_path', '', 'Path to pretrained IQL agent. If empty, trains from scratch.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('am_steps', 200000, 'Number of AM finetuning steps.')
flags.DEFINE_integer('base_train_steps', 1000000, 'Steps for base training (if needed).')
flags.DEFINE_integer('eval_interval', 10000, 'Evaluation interval.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')

# Configurations
def get_full_config():
    # Base Configs (for training from scratch)
    flow = ml_collections.ConfigDict({
        'agent_name': 'flow_bc', 'lr': 3e-4, 'batch_size': 256,
        'actor_hidden_dims': (512, 512, 512, 512), 'actor_layer_norm': False,
        'flow_steps': 10, 'encoder': None,
    })
    iql = ml_collections.ConfigDict({
        'agent_name': 'iql', 'lr': 3e-4, 'batch_size': 256,
        'actor_hidden_dims': (256, 256), 'value_hidden_dims': (256, 256),
        'layer_norm': False, 'actor_layer_norm': False,
        'discount': 0.99, 'tau': 0.005, 'expectile': 0.7, 
        'actor_loss': 'awr', 'alpha': 10.0, 'const_std': True, 'encoder': None,
    })
    
    # Adjoint Matching Config
    am = get_am_config()
    am.actor_hidden_dims = (512, 512, 512, 512)
    
    return ml_collections.ConfigDict({'flow': flow, 'iql': iql, 'am': am})

config_flags.DEFINE_config_dict('config', get_full_config(), 'Full configuration.')

def main(_):
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    # --- 1. Load or Train Base Models ---
    flow_agent = None
    critic_agent = None

    has_pretrained = (FLAGS.pretrained_flow_path != '') and (FLAGS.pretrained_critic_path != '')
    
    if has_pretrained:
        print(f"Loading Pretrained Agents from:\n  {FLAGS.pretrained_flow_path}\n  {FLAGS.pretrained_critic_path}")
        with open(FLAGS.pretrained_flow_path, 'rb') as f:
            flow_agent = pickle.load(f)
        with open(FLAGS.pretrained_critic_path, 'rb') as f:
            critic_agent = pickle.load(f)
    else:
        print("Pretrained paths not provided. Triggering Base Training...")
        flow_agent, critic_agent = train_base_models(
            env_name=FLAGS.env_name,
            seed=FLAGS.seed,
            config=FLAGS.config,
            save_dir=FLAGS.save_dir,
            max_steps=FLAGS.base_train_steps,
            eval_interval=50000 # Evaluate less often during pre-training
        )
        print("Base Training Complete. Proceeding to Finetuning.")

    # --- 2. Initialize Adjoint Matching ---
    print(f"Initializing Adjoint Matching for {FLAGS.env_name}...")
    
    env = make_env(FLAGS.env_name, FLAGS.seed)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 100)
    dataset = Dataset(env.get_dataset(), clip_to_eps=False)
    dataset.seed(FLAGS.seed)
    
    example_batch = dataset.sample(1)
    
    # Inject action_dim into AM config
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
    
    # We might want a baseline score first
    base_score, _ = eval_policy(flow_agent, eval_env, 10)
    print(f"Baseline (BC) Normalized Score: {env.get_normalized_score(base_score)*100:.2f}")

    for i in tqdm.tqdm(range(1, FLAGS.am_steps + 1), smoothing=0.1, desc="AM Finetuning"):
        batch = dataset.sample(FLAGS.config.am.batch_size)
        
        am_agent, info = am_agent.update(batch)
        
        # Logging
        if i % 1000 == 0:
            print(f"Step {i} | AM Loss: {info['loss']:.4f} | Avg Reward (Proxy): {info['avg_reward']:.2f}")
            
        # Evaluation
        if i % FLAGS.eval_interval == 0:
            print("Evaluating AM Policy...")
            eval_score, eval_std = eval_policy(am_agent, eval_env, FLAGS.eval_episodes)
            normalized_score = env.get_normalized_score(eval_score) * 100.0
            
            print(f"Step {i}: Raw Reward: {eval_score:.2f}, Normalized: {normalized_score:.2f}")
            
            if eval_score > best_am_score:
                best_am_score = eval_score
                save_path = os.path.join(FLAGS.save_dir, f'am_finetuned_{FLAGS.env_name}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(am_agent, f)
                print(f"Saved Best AM Model to {save_path}")

if __name__ == '__main__':
    app.run(main)