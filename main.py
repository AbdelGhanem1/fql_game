import glob, tqdm, wandb, os, json, random, time, jax
import gc
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

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
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval (steps per file).')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')
flags.DEFINE_integer('files_per_load', 10, 'Number of dataset files to load into RAM at once.') 

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
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

# --- RETURN LIST OF DATASETS ---
def load_chunk_of_files(env_name, paths, cur_env=None):
    """
    Loads a list of npz files and returns them as a LIST of datasets.
    """
    print(f"Loading chunk of {len(paths)} files...", flush=True)
    start_t = time.time()
    datasets_list = []
    
    # Initialize defaults
    eval_env = None
    val_dataset = None
    
    for i, path in enumerate(paths):
        # We only need to create the env once if it's not provided
        is_first = (cur_env is None) and (i == 0)
        
        if is_first:
            env, eval_env, ds, val_dataset = make_ogbench_env_and_datasets(
                env_name,
                dataset_path=path,
                compact_dataset=False,
            )
            datasets_list.append(ds)
            cur_env = env 
        else:
            # Load data only
            ds, _ = make_ogbench_env_and_datasets(
                env_name,
                dataset_path=path,
                compact_dataset=False,
                dataset_only=True,
                cur_env=cur_env,
            )
            datasets_list.append(ds)

    print(f"Chunk load complete. Loaded {len(datasets_list)} datasets. Time: {time.time() - start_t:.2f}s", flush=True)
        
    return cur_env, eval_env, datasets_list, val_dataset


def main(_):
    exp_name = os.environ.get('WANDB_NAME') or get_exp_name(FLAGS)
    run = setup_wandb(project='qam-reproduce', group=FLAGS.run_group, name=exp_name, tags=FLAGS.tags.split(","))
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    
    dataset_paths = []
    
    # --- HELPER: Process a SINGLE dataset ---
    def process_train_dataset(ds):
        """
        Process a single dataset dictionary.
        """
        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0 and FLAGS.ogbench_dataset_dir is None: 
             new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
             ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if FLAGS.sparse:
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)
        return ds
    # ----------------------------------------

    # data loading
    train_datasets = [] # This will be a LIST of Dataset objects

    if FLAGS.ogbench_dataset_dir is not None:
        assert FLAGS.dataset_replace_interval != 0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]

        if FLAGS.dataset_proportion < 1.:
            num_datasets = len(dataset_paths)
            num_subset_datasets = max(1, int(num_datasets * FLAGS.dataset_proportion))
            print("actual data proportion:", num_subset_datasets / num_datasets)
            dataset_paths = dataset_paths[:num_subset_datasets]

        # --- Initial Chunk Load ---
        batch_paths = []
        files_to_load_now = min(FLAGS.files_per_load, len(dataset_paths))
        
        for i in range(files_to_load_now):
            batch_paths.append(dataset_paths[(dataset_idx + i) % len(dataset_paths)])
        
        # Returns a LIST of raw dicts
        env, eval_env, raw_datasets_list, val_dataset = load_chunk_of_files(
            FLAGS.env_name, 
            batch_paths, 
            cur_env=None
        )
        
        # Process them immediately into a list of Dataset objects
        train_datasets = [process_train_dataset(ds) for ds in raw_datasets_list]
        
        if files_to_load_now >= len(dataset_paths):
            print(f"Loaded all {len(dataset_paths)} available files. Dataset swapping disabled.")
        # --------------------------

    else:
        # Standard Single File Loading
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)
        # Wrap it in a list so the rest of the code works uniformly
        train_datasets = [process_train_dataset(train_dataset)]

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    
    config = FLAGS.agent
    discount = FLAGS.agent.discount
    config["horizon_length"] = FLAGS.horizon_length

    # Use the first loaded dataset for example batch and agent creation
    example_batch = train_datasets[0].sample(())
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    params = agent.network.params
    # filter all target network
    params = {k: v for k, v in params.items() if "target" not in k}

    print(params.keys())
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("param count:", param_count)

    # Setup logging.
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

    logger = LoggingHelper(
        csv_loggers=csv_loggers,
        wandb_logger=wandb,
    )

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

            # --- DATASET MANAGEMENT LOGIC ---
            if FLAGS.ogbench_dataset_dir is not None and \
               FLAGS.dataset_replace_interval != 0:
                
                # 1. Check if we need to load a NEW CHUNK of files from disk
                #    This happens every (files_per_load * replace_interval) steps
                chunk_swap_freq = FLAGS.dataset_replace_interval * FLAGS.files_per_load
                
                # Only swap chunks if we haven't loaded everything
                if FLAGS.files_per_load < len(dataset_paths) and i % chunk_swap_freq == 0 and i > 0:
                    dataset_idx = (dataset_idx + FLAGS.files_per_load) % len(dataset_paths)
                    print(f"Loading new chunk starting at index: {dataset_idx}", flush=True)
                    
                    del train_datasets
                    gc.collect()

                    batch_paths = []
                    for offset in range(FLAGS.files_per_load):
                        batch_paths.append(dataset_paths[(dataset_idx + offset) % len(dataset_paths)])

                    _, _, raw_datasets_list, _ = load_chunk_of_files(
                        FLAGS.env_name,
                        batch_paths,
                        cur_env=env,
                    )
                    train_datasets = [process_train_dataset(ds) for ds in raw_datasets_list]

                # 2. Select the ACTIVE dataset from the currently loaded chunk
                #    We change the active file every 'replace_interval' (1000) steps.
                #    If we have K files in memory, we cycle through 0 to K-1.
                
                # Which step are we at relative to the start of this chunk?
                steps_within_chunk = i % chunk_swap_freq
                
                # Which file index (0 to K-1) corresponds to this step?
                active_file_idx = steps_within_chunk // FLAGS.dataset_replace_interval
                
                # Safety clip (rare edge case on exact boundary)
                active_file_idx = min(active_file_idx, len(train_datasets) - 1)
                
                current_train_dataset = train_datasets[active_file_idx]
            else:
                # Single file mode / non-OGBench
                current_train_dataset = train_datasets[0]
            # --------------------------------

            batch = current_train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)
            
            if config['agent_name'] == 'rebrac':
                agent, offline_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, offline_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                logger.log(offline_info, "offline_agent", step=log_step)

            # eval
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
                
            # saving
            if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
                last_save_path = save_agent(agent, FLAGS.save_dir, log_step)
                save_csv_loggers(csv_loggers, FLAGS.save_dir)

                with open(os.path.join(FLAGS.save_dir, 'progress.tk'), 'w') as f:
                    f.write(f"offline,{i}")

    # transition from offline to online
    if replay_buffer is None:
        # Use the LAST used dataset to initialize buffer structure, or first one
        ref_dataset = train_datasets[0] 
        print(ref_dataset.keys())
        print(ref_dataset["observations"].shape)

        if not FLAGS.balanced_sampling:
            replay_buffer = ReplayBuffer.create_from_initial_dataset(
                dict(ref_dataset), size=ref_dataset.size + FLAGS.online_steps
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

        # --- ONLINE DATASET LOGIC (SAME AS OFFLINE) ---
        if FLAGS.ogbench_dataset_dir is not None and \
           FLAGS.dataset_replace_interval != 0:
            
            chunk_swap_freq = FLAGS.dataset_replace_interval * FLAGS.files_per_load
            
            # 1. Disk Swap Logic
            if FLAGS.files_per_load < len(dataset_paths) and i % chunk_swap_freq == 0 and i > 0:
                dataset_idx = (dataset_idx + FLAGS.files_per_load) % len(dataset_paths)
                print(f"Using new dataset chunk: {dataset_idx}", flush=True)

                del train_datasets
                gc.collect()

                batch_paths = []
                for offset in range(FLAGS.files_per_load):
                    batch_paths.append(dataset_paths[(dataset_idx + offset) % len(dataset_paths)])

                _, _, raw_datasets_list, _ = load_chunk_of_files(
                    FLAGS.env_name,
                    batch_paths,
                    cur_env=env,
                )
                train_datasets = [process_train_dataset(ds) for ds in raw_datasets_list]
                
            # 2. RAM Select Logic
            steps_within_chunk = i % chunk_swap_freq
            active_file_idx = steps_within_chunk // FLAGS.dataset_replace_interval
            active_file_idx = min(active_file_idx, len(train_datasets) - 1)
            current_train_dataset = train_datasets[active_file_idx]

            # 3. Buffer Update Logic (Mimic replacing buffer every 1000 steps)
            if not FLAGS.balanced_sampling:
                # We need to detect if we just switched FILES (every 1000 steps)
                # This could be a file swap within RAM, OR a chunk swap from disk
                if i % FLAGS.dataset_replace_interval == 0:
                    # Overwrite the static part of the buffer with the NEW current file
                    size = current_train_dataset.size
                    for k in current_train_dataset:
                        replay_buffer[k][:size] = current_train_dataset[k][:]
        else:
             current_train_dataset = train_datasets[0]
        # ----------------------------------------------

        # during online rl, the action chunk is executed fully
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
                dataset_batch = current_train_dataset.sample_sequence(config['batch_size'] // 2 * FLAGS.utd_ratio, 
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