import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger
import gc  # <--- CRITICAL IMPORT

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets

from utils.flax_utils import save_agent, restore_agent
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('tags', 'Default', 'Wandb tag.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 500000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 1000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 50000, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/meam.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 10000, 'Dataset replace interval')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')
flags.DEFINE_integer('files_per_load', 5, "Number of dataset files to load and merge at once")

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")
flags.DEFINE_bool('save_last_checkpoint', False, "do not delete the last checkpoint")
flags.DEFINE_bool('save_replay_buffer', False, "do not delete the replay buffer in the end")

flags.DEFINE_bool('balanced_sampling', False, "sample half offline and online replay buffer")

def save_csv_loggers(csv_loggers, save_dir):
    for prefix, csv_logger in csv_loggers.items():
        csv_logger.save(os.path.join(save_dir, f"{prefix}_sv.csv"))

def restore_csv_loggers(csv_loggers, save_dir):
    for prefix, csv_logger in csv_loggers.items():
        if os.path.exists(os.path.join(save_dir, f"{prefix}_sv.csv")):
            csv_logger.restore(os.path.join(save_dir, f"{prefix}_sv.csv"))

def save_buffer_env_state(buffer, env, action_queue, save_dir):
    state = env.get_state()
    env_state = {}
    env_state["env_qpos"] = np.copy(state["qpos"])
    env_state["env_qvel"] = np.copy(state["qvel"])
    if "button_states" in state:
        env_state["env_button_states"] = np.copy(state["button_states"])
    if action_queue is None or len(action_queue) == 0:
        pass
    else:
        env_state["action_queue"] = np.stack(action_queue, axis=0)
    np.savez(os.path.join(save_dir, "buffer.npz"), **buffer, **env_state, pointer=buffer.pointer, size=buffer.size)

def restore_buffer_env_state(restore_path):
    buffer_dict = np.load(os.path.join(restore_path, "buffer.npz"))
    buffer_dict = {k: buffer_dict[k] for k in buffer_dict.files}
    pointer = buffer_dict.pop("pointer")
    size = buffer_dict.pop("size")

    state = {}
    state["qpos"] = buffer_dict.pop("env_qpos")
    state["qvel"] = buffer_dict.pop("env_qvel")
    if "env_button_states" in buffer_dict:
        state["button_states"] = buffer_dict.pop("env_button_states")
    if "action_queue" in buffer_dict:
        state["action_queue"] = buffer_dict.pop("action_queue")
    
    return ReplayBuffer(buffer_dict, pointer=pointer, size=size), state

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

# --- MEMORY SAFE HELPERS ---

def apply_dataset_processing_to_dict(ds_dict):
    """Applies proportion and sparse rewards to a RAW DICTIONARY."""
    if FLAGS.dataset_proportion < 1.0:
        length_key = 'masks' if 'masks' in ds_dict else 'observations'
        total_len = len(ds_dict[length_key])
        new_size = int(total_len * FLAGS.dataset_proportion)
        ds_dict = {k: v[:new_size] for k, v in ds_dict.items()}
    
    if FLAGS.sparse:
        sparse_rewards = (ds_dict["rewards"] != 0.0) * -1.0
        ds_dict["rewards"] = sparse_rewards
        
    return ds_dict

def load_dataset_chunk(env, dataset_paths, start_idx, num_files):
    """Loads raw numpy dicts and merges them."""
    raw_dicts = []
    current_idx = start_idx
    
    # print(f"Loading {num_files} files starting from index {start_idx}...", flush=True)
    
    for _ in range(num_files):
        current_idx = (current_idx + 1) % len(dataset_paths)
        path = dataset_paths[current_idx]
        
        ds_dict, _ = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=path,
            compact_dataset=False,
            dataset_only=True,
            cur_env=env,
        )
        ds_dict = apply_dataset_processing_to_dict(ds_dict)
        raw_dicts.append(ds_dict)

    if num_files == 1:
        merged_data = raw_dicts[0]
    else:
        keys = raw_dicts[0].keys()
        merged_data = {}
        for k in keys:
            merged_data[k] = np.concatenate([d[k] for d in raw_dicts], axis=0)
    
    # print(f"Merged dataset size: {merged_data['observations'].shape[0]}", flush=True)
    return Dataset.create(**merged_data), current_idx

def main(_):
    exp_name = os.environ.get('WANDB_NAME') or get_exp_name(FLAGS)
    run = setup_wandb(project='qam-reproduce', group=FLAGS.run_group, name=exp_name, tags=FLAGS.tags.split(","))
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    
    config = FLAGS.agent
    discount = FLAGS.agent.discount
    config["horizon_length"] = FLAGS.horizon_length

    # Data loading setup
    dataset_idx = 0
    dataset_paths = []
    
    if FLAGS.ogbench_dataset_dir is not None:
        assert FLAGS.dataset_replace_interval != 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]

        if FLAGS.dataset_proportion < 1.:
            num_datasets = len(dataset_paths)
            num_subset_datasets = max(1, int(num_datasets * FLAGS.dataset_proportion))
            print("actual data proportion:", num_subset_datasets / num_datasets)
            dataset_paths = dataset_paths[:num_subset_datasets]

        # Initial Load
        env, eval_env, train_dataset_dict, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
        train_dataset_dict = apply_dataset_processing_to_dict(train_dataset_dict)

        if FLAGS.files_per_load > 1:
            additional_ds, dataset_idx = load_dataset_chunk(
                env, dataset_paths, dataset_idx, FLAGS.files_per_load - 1
            )
            merged_data = {}
            for k in train_dataset_dict:
                merged_data[k] = np.concatenate([train_dataset_dict[k], additional_ds[k]], axis=0)
            train_dataset = Dataset.create(**merged_data)
            # Free memory explicitly
            del additional_ds
            del merged_data
            del train_dataset_dict
            gc.collect()
        else:
            train_dataset = Dataset.create(**train_dataset_dict)
            del train_dataset_dict
            gc.collect()
            
    else:
        env, eval_env, train_dataset_dict, val_dataset = make_env_and_datasets(FLAGS.env_name)
        train_dataset_dict = apply_dataset_processing_to_dict(train_dataset_dict)
        train_dataset = Dataset.create(**train_dataset_dict)
        del train_dataset_dict
        gc.collect()

    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    params = agent.network.params
    params = {k: v for k, v in params.items() if "target" not in k}

    print(params.keys())
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("param count:", param_count)

    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")
    csv_loggers = {prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes}

    last_save_path = None
    if os.path.isdir(FLAGS.save_dir):
        print("trying to load from", FLAGS.save_dir)

        if os.path.exists(os.path.join(FLAGS.save_dir, 'token.tk')):
            exit()

        try:
            with open(os.path.join(FLAGS.save_dir, 'progress.tk'), 'r') as f:
                progress = f.read()
            
            load_stage, load_step = progress.split(",")
            load_step = int(load_step)
            agent = restore_agent(agent, restore_path=FLAGS.save_dir, restore_epoch=load_step)
            restore_csv_loggers(csv_loggers, FLAGS.save_dir)
            if load_stage == "online": 
                replay_buffer, env_state = restore_buffer_env_state(restore_path=FLAGS.save_dir)
            else:
                replay_buffer, env_state = None, None
            success = True
        except:
            success = False
            load_stage = None
            load_step = None
            replay_buffer = None
    else:
        success = False
        load_stage = None
        load_step = None
        replay_buffer = None
    
    if not success: 
        print("failed to load prev run")
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        flag_dict = get_flag_dict()
        with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
            json.dump(flag_dict, f)

    logger = LoggingHelper(csv_loggers=csv_loggers, wandb_logger=wandb)

    # Offline RL
    offline_init_time = time.time()
    if load_stage is not None and load_stage == "online":
        print("skipping offline")
    else:
        if load_stage == "offline" and load_step is not None:
            start_step = load_step + 1
            print(f"restoring from offline step {start_step}")
        else:
            start_step = 1
            
        for i in tqdm.tqdm(range(start_step, FLAGS.offline_steps + 1)):
            log_step = i
            
            # --- CRITICAL FIX: Dataset Replacement Logic ---
            load_interval = FLAGS.dataset_replace_interval * FLAGS.files_per_load
            
            if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % load_interval == 0:
                print(f"Replacing dataset at step {i}. Cleaning memory...", flush=True)
                
                # 1. Delete old dataset
                del train_dataset
                
                # 2. Force Garbage Collection to free the memory
                gc.collect() 
                
                # 3. Load new dataset
                train_dataset, dataset_idx = load_dataset_chunk(
                    env, 
                    dataset_paths, 
                    dataset_idx, 
                    FLAGS.files_per_load
                )

            batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)
            
            if config['agent_name'] == 'rebrac':
                agent, offline_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, offline_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                logger.log(offline_info, "offline_agent", step=log_step)

            if i == FLAGS.offline_steps or \
                (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
                eval_info, _, _ = evaluate(
                    agent=agent,
                    env=eval_env,
                    action_dim=example_batch["actions"].shape[-1],
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                )
                logger.log(eval_info, "eval", step=log_step)
                
            if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
                last_save_path = save_agent(agent, FLAGS.save_dir, log_step)
                save_csv_loggers(csv_loggers, FLAGS.save_dir)

                with open(os.path.join(FLAGS.save_dir, 'progress.tk'), 'w') as f:
                    f.write(f"offline,{i}")

    # transition from offline to online
    if replay_buffer is None:
        if not FLAGS.balanced_sampling:
            replay_buffer = ReplayBuffer.create_from_initial_dataset(
                dict(train_dataset), size=train_dataset.size + FLAGS.online_steps
            )
        else:
            replay_buffer = ReplayBuffer.create(example_batch, size=FLAGS.online_steps)
    
    action_dim = example_batch["actions"].shape[-1]

    # Online RL
    update_info = {}
    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()

    if load_stage == "online" and load_step is not None and env_state is not None:
        start_step = load_step + 1
        if "action_queue" in env_state:
            action_queue = list(np.reshape(env_state.pop("action_queue"), (-1, action_dim)))
            print("restored action queue:", action_queue)
        else:
            action_queue = []
        ob, info = env.reset(options={"set_state": env_state})
        print(f"restoring from online step {start_step}")
    else:
        action_queue = []
        ob, _ = env.reset()
        start_step = 1

    for i in tqdm.tqdm(range(start_step, FLAGS.online_steps + 1)):
        log_step = FLAGS.offline_steps + i
        online_rng, key = jax.random.split(online_rng)

        # --- CRITICAL FIX: Online Loading Logic ---
        load_interval = FLAGS.dataset_replace_interval * FLAGS.files_per_load
        
        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % load_interval == 0:
            print(f"Replacing dataset at online step {i}. Cleaning memory...", flush=True)
            
            # Explicit GC
            del train_dataset
            gc.collect()

            train_dataset, dataset_idx = load_dataset_chunk(
                env, 
                dataset_paths, 
                dataset_idx, 
                FLAGS.files_per_load
            )
            
            if not FLAGS.balanced_sampling:
                copy_size = min(train_dataset.size, replay_buffer.size)
                for k in train_dataset:
                    replay_buffer[k][:copy_size] = train_dataset[k][:copy_size]
        
        if len(action_queue) == 0:
            if FLAGS.balanced_sampling and i < FLAGS.start_training:
                action = np.random.rand(action_dim) * 2. - 1.
                action = np.clip(action, -1., 1.)
            else:
                action = agent.sample_actions(observations=ob, rng=key)
            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        logger.log(env_info, "env", step=log_step)

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        if done:
            ob, _ = env.reset()
            action_queue = []
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            if FLAGS.balanced_sampling:
                dataset_batch = train_dataset.sample_sequence(config['batch_size'] // 2 * FLAGS.utd_ratio, 
                        sequence_length=FLAGS.horizon_length, discount=discount)
                replay_batch = replay_buffer.sample_sequence(FLAGS.utd_ratio * config['batch_size'] // 2, 
                    sequence_length=FLAGS.horizon_length, discount=discount)
                
                batch = {k: np.concatenate([
                    dataset_batch[k].reshape((FLAGS.utd_ratio, config["batch_size"] // 2) + dataset_batch[k].shape[1:]), 
                    replay_batch[k].reshape((FLAGS.utd_ratio, config["batch_size"] // 2) + replay_batch[k].shape[1:])], axis=1) for k in dataset_batch}
            else:
                batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
                            sequence_length=FLAGS.horizon_length, discount=discount)
                batch = jax.tree.map(lambda x: x.reshape((
                    FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            if config['agent_name'] == 'rebrac':
                agent, update_info["online_agent"] = agent.batch_update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            last_save_path = save_agent(agent, FLAGS.save_dir, log_step)
            save_buffer_env_state(replay_buffer, env, action_queue, FLAGS.save_dir)
            save_csv_loggers(csv_loggers, FLAGS.save_dir)
            with open(os.path.join(FLAGS.save_dir, 'progress.tk'), 'w') as f:
                f.write(f"online,{i}")
            print("saved action queue:", action_queue)
            print("saved buffer:", i, replay_buffer.pointer, replay_buffer.size)

    end_time = time.time()
    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qvel"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

    all_files = os.listdir(FLAGS.save_dir)
    for relative_path in all_files:
        full_path = os.path.join(FLAGS.save_dir, relative_path)
        if os.path.isfile(full_path):
            if relative_path.startswith("params"):
                if not FLAGS.save_last_checkpoint or relative_path != last_save_path:
                    print(f"removing {full_path}")
                    os.remove(full_path)
            if relative_path == "buffer.npz" and not FLAGS.save_replay_buffer:
                os.remove(full_path)

    wandb.finish()

if __name__ == '__main__':
    app.run(main)