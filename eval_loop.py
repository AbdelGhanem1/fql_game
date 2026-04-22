import glob, tqdm, wandb, os, re, jax
from absl import app, flags
from ml_collections import config_flags
import numpy as np

# Adjust imports based on your repository structure
from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from utils.flax_utils import restore_agent
from utils.datasets import Dataset
from evaluation import evaluate
from agents import agents

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

# Basic environment and run flags
flags.DEFINE_string('run_group', 'Eval_Group', 'Run group.')
flags.DEFINE_string('tags', 'Eval', 'Wandb tag.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'puzzle-4x4-play-singletask-task5-v0', 'Environment name.')
flags.DEFINE_string('save_dir', '', 'Exact directory where params_* files are saved.')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

# Evaluation flags
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('horizon_length', 5, 'Action chunking length.')
flags.DEFINE_bool('sparse', False, "Make the task sparse reward")

config_flags.DEFINE_config_file('agent', 'agents/meam.py', lock_config=False)

def main(_):
    # 1. Initialize WandB explicitly for the new project
    wandb_name = os.environ.get('WANDB_NAME', f"eval_{FLAGS.env_name}")
    wandb_project = os.environ.get('WANDB_PROJECT', 'qam-reproduce-eval')
    
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        group=FLAGS.run_group,
        tags=FLAGS.tags.split(","),
        config=FLAGS
    )

    # 2. Setup Environment and Dataset (needed for agent initialization)
    if FLAGS.ogbench_dataset_dir is not None:
        dataset_paths = [file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file]
        env, eval_env, train_dataset, _ = make_ogbench_env_and_datasets(
            FLAGS.env_name, dataset_path=dataset_paths[0], compact_dataset=False
        )
    else:
        env, eval_env, train_dataset, _ = make_env_and_datasets(FLAGS.env_name)

    # Reformat dataset to get example batch for agent creation
    ds = Dataset.create(**train_dataset)
    example_batch = ds.sample(())
    action_dim = example_batch["actions"].shape[-1]

    # 3. Create Agent structure
    config = FLAGS.agent
    config["horizon_length"] = FLAGS.horizon_length
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # 4. Find all saved checkpoints in the save_dir
    if not os.path.isdir(FLAGS.save_dir):
        print(f"Error: Target directory does not exist:\n{FLAGS.save_dir}")
        wandb.finish()
        return

    checkpoints = []
    for f in os.listdir(FLAGS.save_dir):
        # UPDATED REGEX: Looks for 'params_' followed by digits, ignoring any extensions like .pkl
        match = re.search(r'^params_(\d+)', f)
        if match:
            checkpoints.append(int(match.group(1)))
            
    checkpoints = sorted(list(set(checkpoints)))
    
    if not checkpoints:
        print(f"No 'params_*' checkpoints found in:\n{FLAGS.save_dir}")
        wandb.finish()
        return

    print(f"Found {len(checkpoints)} checkpoints to evaluate in {FLAGS.save_dir}")

    # 5. Iterate and Evaluate
    for step in tqdm.tqdm(checkpoints, desc="Evaluating Checkpoints"):
        try:
            # We pass FLAGS.save_dir explicitly, and restore_agent handles appending the prefix
            current_agent = restore_agent(agent, restore_path=FLAGS.save_dir, restore_epoch=step)
            
            # Run evaluation
            eval_info, _, _ = evaluate(
                agent=current_agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            
            # Log metrics to WandB tied to the actual training step
            wandb.log({f"eval/{k}": v for k, v in eval_info.items()}, step=step)
            print(f"Step {step} evaluation complete. Metrics: {eval_info}")
            
        except Exception as e:
            print(f"Failed to evaluate checkpoint at step {step}: {e}")

    wandb.finish()

if __name__ == '__main__':
    app.run(main)
