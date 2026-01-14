import os
import pickle
import numpy as np
import jax
import ml_collections
from absl import app, flags
from ml_collections import config_flags
import tqdm
import flax # Required for serialization

from envs.env_utils import make_env_and_datasets 
from utils.datasets import Dataset
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

    def get_default_config():
        flow = ml_collections.ConfigDict({
            'agent_name': 'flow_bc', 'lr': 3e-4, 'batch_size': 256,
            'actor_hidden_dims': (512, 512, 512, 512), 'actor_layer_norm': False,
            'flow_steps': 10, 'encoder': ml_collections.config_dict.placeholder(str),
            'action_dim': ml_collections.config_dict.placeholder(int),
        })
        iql = ml_collections.ConfigDict({
            'agent_name': 'iql', 'lr': 3e-4, 'batch_size': 256,
            'actor_hidden_dims': (256, 256), 'value_hidden_dims': (256, 256),
            'layer_norm': True, 'actor_layer_norm': False,
            'discount': 0.99, 'tau': 0.005, 'expectile': 0.7,
            'actor_loss': 'awr', 'alpha': 10.0, 'const_std': True,
            'encoder': ml_collections.config_dict.placeholder(str),
        })
        return ml_collections.ConfigDict({'flow': flow, 'iql': iql})

    config_flags.DEFINE_config_dict('config', get_default_config(), 'Combined configuration.')

def eval_policy(agent, env, num_episodes=10):
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.sample_actions(obs[None, :], seed=jax.random.PRNGKey(0))
            action = np.array(action[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns), np.std(returns)

def train_base_models(env_name, seed, config, save_dir, max_steps=1000000, eval_interval=50000):
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
    
    for i in tqdm.tqdm(range(1, max_steps + 1), smoothing=0.1, desc="Base Training"):
        batch = train_dataset.sample(config.flow.batch_size)
        
        flow_agent, flow_info = flow_agent.update(batch)
        critic_agent, critic_info = critic_agent.update(batch)

        if i % 5000 == 0:
            print(f"Step {i} | Flow Loss: {flow_info['loss']:.4f} | Q: {critic_info['critic/critic_loss']:.4f}")

        if i % eval_interval == 0:
            eval_score, _ = eval_policy(flow_agent, eval_env, 2)
            try:
                norm_score = env.get_normalized_score(eval_score) * 100.0
            except AttributeError:
                norm_score = eval_score
            
            print(f"Step {i} Eval: {norm_score:.2f}")
            
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_agents['flow'] = flow_agent
                best_agents['critic'] = critic_agent
                
                # [CRITICAL FIX] Save state_dict, not object
                with open(os.path.join(save_dir, f'base_flow_{env_name}.pkl'), 'wb') as f:
                    pickle.dump(flax.serialization.to_state_dict(flow_agent), f)
                with open(os.path.join(save_dir, f'base_critic_{env_name}.pkl'), 'wb') as f:
                    pickle.dump(flax.serialization.to_state_dict(critic_agent), f)

    return best_agents['flow'], best_agents['critic']

def main(_):
    train_base_models(FLAGS.env_name, FLAGS.seed, FLAGS.config, FLAGS.save_dir, FLAGS.max_steps, FLAGS.eval_interval)

if __name__ == '__main__':
    app.run(main)