#!/bin/bash

# Usage: ./run_docker.sh [JOB_INDEX]
JOB_INDEX=${1:-0}

# ==============================================================================
# 1. PARAMETER SELECTION
# ==============================================================================
SEEDS=(40004 10001 20002 50005 60006 80008 70007 30003)
TASKS=(1 2 3 4 5)

# --- QAM-E SPECIFIC ---
INV_TEMPS=(10.0)    # Optimal inv_temp for QAM-E on antmaze-giant
EDIT_SCALES=(0.1)   # Activates QAM-E

# --- DOMAIN SPECIFIC: antmaze-giant ---
DISCOUNTS=(0.995)   # Long-horizon discount factor

NUM_SEEDS=${#SEEDS[@]}
NUM_TASKS=${#TASKS[@]}
NUM_INV_TEMPS=${#INV_TEMPS[@]}
NUM_EDIT_SCALES=${#EDIT_SCALES[@]}
NUM_DISCOUNTS=${#DISCOUNTS[@]}

# Indexing Logic
TASK_ID=${TASKS[$(( JOB_INDEX % NUM_TASKS ))]}
SEED=${SEEDS[$(( (JOB_INDEX / NUM_TASKS) % NUM_SEEDS ))]}
INV_TEMP=${INV_TEMPS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS)) % NUM_INV_TEMPS ))]}
EDIT_SCALE=${EDIT_SCALES[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_INV_TEMPS)) % NUM_EDIT_SCALES ))]}
DISCOUNT=${DISCOUNTS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_INV_TEMPS * NUM_EDIT_SCALES)) % NUM_DISCOUNTS ))]}

echo "=========================================="
echo "Cloud-Hardened Job Index: $JOB_INDEX"
echo "Config: Antmaze-Giant Task=$TASK_ID | Seed=$SEED"
echo "Params: QAM-E | InvTemp=$INV_TEMP | EditScale=$EDIT_SCALE"
echo "=========================================="

# ==============================================================================
# 2. DOCKER ENVIRONMENT SETUP
# ==============================================================================
CONDA_ENV="/opt/conda/envs/fql_env"
SITE_PACKAGES="$CONDA_ENV/lib/python3.10/site-packages"
PYTHON_EXEC="$CONDA_ENV/bin/python"

export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$SITE_PACKAGES/nvidia/cuda_runtime/../.. --xla_gpu_strict_conv_algorithm_picker=false"

export NVIDIA_TF32_OVERRIDE=1
export JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export JAX_ENABLE_X64=False

# ==============================================================================
# 3. PATHS & TRAINING
# ==============================================================================
PROJECT_DIR="/models/fql_game"

# CLOUD ARMOR: Pointing to the physical RAM disk
DATASET_DIR="/dev/shm/antmaze-giant" 

# CLOUD ARMOR: Ensure the dataset is actually in RAM before starting
if [ ! -d "$DATASET_DIR" ]; then
    echo "Loading dataset into RAM disk..."
    cp -r /workspace/datasets/antmaze-giant /dev/shm/
fi

cd "$PROJECT_DIR"
mkdir -p /models/logs /models/saved_models

export WANDB_MODE="offline"

export WANDB_PROJECT="qam_e_antmaze_giant_n_step"
export WANDB_NAME="task${TASK_ID}_invTemp${INV_TEMP}_edit${EDIT_SCALE}_seed${SEED}"

echo "🚀 Starting Cloud Training with NUMA Pinning..."

# CLOUD ARMOR: taskset -c 0-31 locks the CPU threads to Socket 0
taskset -c 0-31 "$PYTHON_EXEC" main.py \
    --run_group=antmaze-giant_Docker_QAM_E \
    --agent=agents/qam.py \
    --seed=${SEED} \
    --env_name=antmaze-giant-navigate-singletask-task${TASK_ID}-v0 \
    --ogbench_dataset_dir="${DATASET_DIR}" \
    --sparse=False \
    --horizon_length=5 \
    --agent.action_chunking=False \
    --balanced_sampling=False \
    --agent.discount=${DISCOUNT} \
    --agent.inv_temp=${INV_TEMP} \
    --agent.edit_scale=${EDIT_SCALE} \
    --agent.fql_alpha=0.0 \
    --agent.flow_steps=10 \
    --agent.num_qs=10 \
    --agent.rho=0.5 \
    --agent.batch_size=256 \
    --offline_steps=1000000 \
    --online_steps=0 \
    --eval_interval=50000 \
    --save_interval=500000 \
    --dataset_replace_interval=2000000 \
    --save_dir="/models/saved_models/job_${JOB_INDEX}_qam_e_antmaze_giant_task${TASK_ID}_invTemp${INV_TEMP}"

echo "✅ Job $JOB_INDEX complete!"
