import os

if 'MPLBACKEND' in os.environ:
    del os.environ['MPLBACKEND']
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
from collections import deque
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

# [NEW] Uncertainty Parameter
flags.DEFINE_float('uncertainty_beta', 2.0, 'LCB Beta: Penalty for ensemble disagreement.')

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

def log_reward_comparison(base_trajs, finetuned_trajs, wandb_key="comparison/reward_profile"):
    def get_curve_stats(trajs):
        all_rewards = [t['reward'] for t in trajs]
        max_len = max(len(r) for r in all_rewards)
        padded_rewards = np.full((len(all_rewards), max_len), np.nan)
        for i, r in enumerate(all_rewards):
            padded_rewards[i, :len(r)] = r
        mean = np.nanmean(padded_rewards, axis=0)
        std = np.nanstd(padded_rewards, axis=0)
        steps = np.arange(max_len)
        return steps, mean, std

    b_steps, b_mean, b_std = get_curve_stats(base_trajs)
    f_steps, f_mean, f_std = get_curve_stats(finetuned_trajs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(b_steps, b_mean, label=f'Base Model (Area={np.nansum(b_mean):.1f})', color='blue', alpha=0.8)
    ax.fill_between(b_steps, b_mean - b_std, b_mean + b_std, color='blue', alpha=0.15)
    ax.plot(f_steps, f_mean, label=f'Finetuned AM (Area={np.nansum(f_mean):.1f})', color='red', alpha=0.8)
    ax.fill_between(f_steps, f_mean - f_std, f_mean + f_std, color='red', alpha=0.15)
    
    ax.set_title("Reward Profile: Base vs Finetuned")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Step Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    wandb.log({wandb_key: wandb.Image(fig)})
    plt.close(fig)

def main(_):
    set_global_seed(FLAGS.seed)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    wandb.init(config=FLAGS.config.to_dict())
    
    env, eval_env, train_dataset_dict, _ = make_env_and_datasets(FLAGS.env_name, frame_stack=None)
    train_dataset = Dataset.create(**train_dataset_dict)
    example_batch = train_dataset.sample(1)
    
    flow_agent = None
    critic_agent = None
    
    seed_base_flow_path = os.path.join(FLAGS.save_dir, f'base_flow_{FLAGS.env_name}_seed{FLAGS.seed}.pkl')
    # Use a different name if retraining with 10 ensembles to avoid confusion
    seed_base_critic_path = os.path.join(FLAGS.save_dir, f'base_critic_ens10_{FLAGS.env_name}_seed{FLAGS.seed}.pkl')

    if FLAGS.pretrained_flow_path != '':
        load_flow = FLAGS.pretrained_flow_path
        load_critic = FLAGS.pretrained_critic_path
        print(f"[Seed {FLAGS.seed}] Loading provided Base Agents:\n  - {load_flow}")
    elif os.path.exists(seed_base_critic_path):
        load_flow = seed_base_flow_path
        load_critic = seed_base_critic_path
        print(f"[Seed {FLAGS.seed}] Found existing Base Agents! Skipping Base Training.\n  - {load_critic}")
    else:
        load_flow = None
        load_critic = None

    if load_flow and load_critic:
        with open(load_flow, 'rb') as f:
            flow_state = pickle.load(f)
            flow_agent = FlowBCAgent.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], FLAGS.config.flow)
            flow_agent = flax.serialization.from_state_dict(flow_agent, flow_state)
        with open(load_critic, 'rb') as f:
            critic_state = pickle.load(f)
            critic_agent = IQLAgent.create(FLAGS.seed, example_batch['observations'], example_batch['actions'], FLAGS.config.iql)
            critic_agent = flax.serialization.from_state_dict(critic_agent, critic_state)
    else:
        print(f"[Seed {FLAGS.seed}] No valid model found. Starting Base Training (Ensemble=10)...")
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
    FLAGS.config.am.uncertainty_beta = FLAGS.uncertainty_beta # [NEW]

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
    base_actions = am_agent.sample_actions(monitor_obs, seed=jax.random.PRNGKey(0), temperature=0.0)
    
    # Check baseline stats
    qs_base = critic_agent.network.select('target_critic')(monitor_obs, base_actions)
    if isinstance(qs_base, (list, tuple)):
        qs_base = jnp.stack(qs_base, axis=0)
        mean_base_q = jnp.mean(jnp.min(qs_base, axis=0)) # Conservative baseline
    else:
        mean_base_q = jnp.mean(qs_base)
    
    wandb.log({"monitor/baseline_q": mean_base_q})

    # --- Evaluate Base Agent ---
    print(f"[Seed {FLAGS.seed}] Profiling Base Agent...")
    base_stats, base_trajs, _ = evaluate(
        agent=flow_agent, 
        env=eval_env, 
        num_eval_episodes=FLAGS.eval_episodes, 
        eval_temperature=FLAGS.eval_temperature
    )
    wandb.log({"base/final_norm_score": base_stats.get('episode.normalized_return', 0)})

    # --- 5. Finetuning Loop ---
    print(f"[Seed {FLAGS.seed}] Starting AM Finetuning (Beta: {FLAGS.uncertainty_beta})...")
    
    pbar = tqdm.tqdm(range(1, FLAGS.am_steps + 1), smoothing=0.1, desc=f"AM-B{FLAGS.uncertainty_beta}", mininterval=5.0, ncols=100)
    score_history = deque(maxlen=10)

    for i in pbar:
        batch = train_dataset.sample(FLAGS.config.am.batch_size)
        am_agent, info = am_agent.update(batch)
        
        if i % FLAGS.am_eval_interval == 0:
            # Monitor Metrics
            current_actions = am_agent.sample_actions(monitor_obs, seed=jax.random.PRNGKey(0), temperature=0.0)
            qs_curr = critic_agent.network.select('target_critic')(monitor_obs, current_actions)
            
            if isinstance(qs_curr, (list, tuple)):
                qs_curr_stack = jnp.stack(qs_curr, axis=0)
                mean_curr_q = jnp.mean(jnp.min(qs_curr_stack, axis=0))
                std_curr_q = jnp.mean(jnp.std(qs_curr_stack, axis=0)) # Average uncertainty
            else:
                mean_curr_q = jnp.mean(qs_curr)
                std_curr_q = 0.0

            q_improvement = mean_curr_q - mean_base_q
            action_drift = float(jnp.mean(jnp.linalg.norm(current_actions - base_actions, axis=-1)))
            
            wandb.log({
                "am/loss": info['loss'],
                "monitor/delta_q": q_improvement,
                "monitor/action_drift": action_drift,
                "monitor/uncertainty_std": std_curr_q,
                "am_step": i
            })

            # Run Evaluation
            metrics, _, _ = evaluate(am_agent, eval_env, num_eval_episodes=FLAGS.eval_episodes, eval_temperature=FLAGS.eval_temperature)
            eval_score = metrics.get('episode.return', metrics.get('evaluation/return', -1000))
            
            try: normalized_score = env.unwrapped.get_normalized_score(eval_score) * 100.0
            except: normalized_score = eval_score
            
            score_history.append(normalized_score)
            ma_score = np.mean(score_history)

            wandb.log({
                "am/raw_score": eval_score,
                "am/norm_score": normalized_score,
                "am/norm_score_ma": ma_score,
                "am_step": i
            })
            
            pbar.set_description(f"AM (Drift:{action_drift:.2f}|Score:{normalized_score:.1f}|MA:{ma_score:.1f})")

    # --- Evaluate Final Agent ---
    print(f"[Seed {FLAGS.seed}] Profiling Final Agent...")
    final_stats, final_trajs, _ = evaluate(
        agent=am_agent, 
        env=eval_env, 
        num_eval_episodes=FLAGS.eval_episodes, 
        eval_temperature=FLAGS.eval_temperature
    )

    log_reward_comparison(base_trajs, final_trajs)
    print("Comparison plot uploaded to WandB.")

if __name__ == '__main__':
    app.run(main)