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

FLAGS = flags.FLAGS
if __name__ == '__main__':
    flags.DEFINE_string('env_name', 'halfcheetah-medium-v2', 'Environment name.')
    flags.DEFINE_string('save_dir', './saved_models/', 'Directory to save checkpoints.')
    flags.DEFINE_integer('seed', 42, 'Random seed.')
    flags.DEFINE_integer('eval_episodes', 10, 'Number of evaluation episodes.')
    flags.DEFINE_integer('max_steps', 1000000, 'Number of training steps.')
    flags.DEFINE_integer('log_interval', 5000, 'Log interval.')
    flags.DEFINE_integer('eval_interval', 50000, 'Evaluation interval.')
    flags.DEFINE_integer('batch_size', 256, 'Batch size.')
    flags.DEFINE_float('eval_temperature', 1.0, 'Temperature for evaluation.')

    def get_default_config():
        flow = ml_collections.ConfigDict({
            'agent_name': 'flow_bc', 'lr': 3e-4, 'batch_size': 256,
            # [FIX] Deeper network for complex tasks
            'actor_hidden_dims': (256, 256, 256, 256), 
            'actor_layer_norm': False,
            'flow_steps': 10, 'encoder': ml_collections.config_dict.placeholder(str),
            'action_dim': ml_collections.config_dict.placeholder(int),
        })
        iql = ml_collections.ConfigDict({
            'agent_name': 'iql', 'lr': 3e-4, 'batch_size': 256,
            # [FIX] Deeper network + Layer Norm enabled
            'actor_hidden_dims': (256, 256, 256, 256), 
            'value_hidden_dims': (256, 256, 256, 256),
            'layer_norm': True, 
            'actor_layer_norm': False,
            'discount': 0.99, 
            'tau': 0.005, 
            'expectile': 0.7,
            'actor_loss': 'awr', 
            'alpha': 10.0, 
            'const_std': True,
            'encoder': ml_collections.config_dict.placeholder(str),
        })
        return ml_collections.ConfigDict({'flow': flow, 'iql': iql})

    config_flags.DEFINE_config_dict('config', get_default_config(), 'Combined configuration.')

def train_base_models(env_name, seed, config, save_dir, max_steps=1000000, eval_interval=50000, eval_episodes=10, eval_temperature=1.0):
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[Base Training] Initializing {env_name}...")
    env, eval_env, train_dataset_dict, val_dataset = make_env_and_datasets(env_name, frame_stack=None)
    train_dataset = Dataset.create(**train_dataset_dict)
    
    print(f"[Base Training] Creating Agents...")
    example_batch = train_dataset.sample(1)
    
    flow_agent = FlowBCAgent.create(seed, example_batch['observations'], example_batch['actions'], config.flow)
    critic_agent = IQLAgent.create(seed, example_batch['observations'], example_batch['actions'], config.iql)

    print("[Base Training] Starting Loop...")
    best_eval_score = -float('inf')
    best_agents = {'flow': flow_agent, 'critic': critic_agent}
    
    pbar = tqdm.tqdm(range(1, max_steps + 1), smoothing=0.1, desc="Base Training", mininterval=5.0, ncols=100)
    
    for i in pbar:
        batch = train_dataset.sample(config.flow.batch_size)
        
        flow_agent, flow_info = flow_agent.update(batch)
        critic_agent, critic_info = critic_agent.update(batch)

        if i % 5000 == 0:
            pbar.set_description(f"Base Training (FlowL: {flow_info['loss']:.3f} | QL: {critic_info['critic/critic_loss']:.3f})")

        if i % eval_interval == 0:
            print(f"\n--- Eval Triggered at Step {i} ---")
            print(f"    Train Metrics > Flow Loss: {flow_info['loss']:.4f} | Q Loss: {critic_info['critic/critic_loss']:.4f}")
            
            # --- 1. Evaluate Flow Agent ---
            flow_metrics, _, _ = evaluate(flow_agent, eval_env, config.flow, eval_episodes, eval_temperature=eval_temperature)
            flow_raw = flow_metrics.get('episode.return', flow_metrics.get('evaluation/return', -1000))
            try: flow_norm = env.unwrapped.get_normalized_score(flow_raw) * 100.0
            except: flow_norm = flow_raw

            # --- 2. Evaluate IQL Agent (Critic) ---
            # IQL agent has a sample_actions method (AWR), so we can eval it directly
            critic_metrics, _, _ = evaluate(critic_agent, eval_env, config.iql, eval_episodes, eval_temperature=eval_temperature)
            critic_raw = critic_metrics.get('episode.return', critic_metrics.get('evaluation/return', -1000))
            try: critic_norm = env.unwrapped.get_normalized_score(critic_raw) * 100.0
            except: critic_norm = critic_raw

            print(f"    Flow Agent    > Raw: {flow_raw:.2f} | Norm: {flow_norm:.2f}")
            print(f"    IQL Agent     > Raw: {critic_raw:.2f} | Norm: {critic_norm:.2f}")
            print(f"--------------------------------\n")
            
            # Save if Flow Agent improves (Primary Goal)
            if flow_raw > best_eval_score:
                best_eval_score = flow_raw
                best_agents['flow'] = flow_agent
                best_agents['critic'] = critic_agent
                
                with open(os.path.join(save_dir, f'base_flow_{env_name}.pkl'), 'wb') as f:
                    pickle.dump(flax.serialization.to_state_dict(flow_agent), f)
                with open(os.path.join(save_dir, f'base_critic_{env_name}.pkl'), 'wb') as f:
                    pickle.dump(flax.serialization.to_state_dict(critic_agent), f)

    return best_agents['flow'], best_agents['critic']

def main(_):
    train_base_models(FLAGS.env_name, FLAGS.seed, FLAGS.config, FLAGS.save_dir, FLAGS.max_steps, FLAGS.eval_interval, FLAGS.eval_episodes, FLAGS.eval_temperature)

if __name__ == '__main__':
    app.run(main)