#!/bin/bash

# Usage: ./run_docker.sh [JOB_INDEX]
JOB_INDEX=${1:-0}

# ==============================================================================
# 1. PARAMETER SELECTION
# ==============================================================================
SEEDS=(40004 10001 20002 50005)
TASKS=(4 1)

ALPHAS=(0.2)        # ME_AM_ALPHA
TEMPS=(0.8)         # INV_TEMP
TAU_SCORES=(0.001)  # TAU_SCORE

TAU_CRITICS=(5.0 3.0)   
MIXTURES=(0.0)      # MIXTURE_PROB
DISCOUNTS=(0.995)   

SCORE_MODES=("fast")
HIDDEN_DIMS=("[512,512,512,512]")

NUM_SEEDS=${#SEEDS[@]}
NUM_TASKS=${#TASKS[@]}
NUM_ALPHAS=${#ALPHAS[@]}
NUM_TEMPS=${#TEMPS[@]}
NUM_TAU_SCORES=${#TAU_SCORES[@]}
NUM_TAU_CRITICS=${#TAU_CRITICS[@]}
NUM_MIXTURES=${#MIXTURES[@]}
NUM_DISCOUNTS=${#DISCOUNTS[@]}
NUM_MODES=${#SCORE_MODES[@]}
NUM_DIMS=${#HIDDEN_DIMS[@]}

# Indexing Logic
TASK_ID=${TASKS[$(( JOB_INDEX % NUM_TASKS ))]}
SEED=${SEEDS[$(( (JOB_INDEX / NUM_TASKS) % NUM_SEEDS ))]}
INV_TEMP=${TEMPS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS)) % NUM_TEMPS ))]}
ME_AM_ALPHA=${ALPHAS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS)) % NUM_ALPHAS ))]}
MIXTURE_PROB=${MIXTURES[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS * NUM_ALPHAS)) % NUM_MIXTURES ))]}
SCORE_MODE=${SCORE_MODES[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS * NUM_ALPHAS * NUM_MIXTURES)) % NUM_MODES ))]}
CURRENT_DIMS=${HIDDEN_DIMS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS * NUM_ALPHAS * NUM_MIXTURES * NUM_MODES)) % NUM_DIMS ))]}
TAU_CRITIC=${TAU_CRITICS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS * NUM_ALPHAS * NUM_MIXTURES * NUM_MODES * NUM_DIMS)) % NUM_TAU_CRITICS ))]}
TAU_SCORE=${TAU_SCORES[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS * NUM_ALPHAS * NUM_MIXTURES * NUM_MODES * NUM_DIMS * NUM_TAU_CRITICS)) % NUM_TAU_SCORES ))]}
DISCOUNT=${DISCOUNTS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS * NUM_ALPHAS * NUM_MIXTURES * NUM_MODES * NUM_DIMS * NUM_TAU_CRITICS * NUM_TAU_SCORES)) % NUM_DISCOUNTS ))]}

DIMS_TAG=$(echo $CURRENT_DIMS | tr -d '[],')

echo "=========================================="
echo "Cloud-Hardened Job Index: $JOB_INDEX"
echo "Config: Humanoidmaze-Large Task=$TASK_ID | Seed=$SEED | Mode=$SCORE_MODE | Dims=$DIMS_TAG"
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
DATASET_DIR="/dev/shm/humanoidmaze-large" 

# CLOUD ARMOR: Ensure the dataset is actually in RAM before starting
if [ ! -d "$DATASET_DIR" ]; then
    echo "Loading dataset into RAM disk..."
    cp -r /workspace/datasets/humanoidmaze-large /dev/shm/
fi

cd "$PROJECT_DIR"
mkdir -p /models/logs /models/saved_models

export WANDB_MODE="offline"

export WANDB_PROJECT="humanoidmaze-large_mirror_descent"
export WANDB_NAME="task${TASK_ID}_tmp${INV_TEMP}_mprob${MIXTURE_PROB}_${SCORE_MODE}_dims${DIMS_TAG}_alpha${ME_AM_ALPHA}_tauC${TAU_CRITIC}_tauS${TAU_SCORE}_seed${SEED}"

echo "🚀 Starting Cloud Training with NUMA Pinning..."

# CLOUD ARMOR: taskset -c 0-31 locks the CPU threads to Socket 0
taskset -c 0-31 "$PYTHON_EXEC" main.py \
    --run_group=humanoidmaze-large_Docker_Repro \
    --agent=agents/meam.py \
    --seed=${SEED} \
    --env_name=humanoidmaze-large-navigate-singletask-task${TASK_ID}-v0 \
    --ogbench_dataset_dir="${DATASET_DIR}" \
    --sparse=False \
    --horizon_length=5 \
    --agent.action_chunking=False \
    --balanced_sampling=False \
    --agent.discount=${DISCOUNT} \
    --agent.inv_temp=${INV_TEMP} \
    --agent.mixture_prob=${MIXTURE_PROB} \
    --agent.me_am_alpha=${ME_AM_ALPHA} \
    --agent.tau_critic=${TAU_CRITIC} \
    --agent.tau_score=${TAU_SCORE} \
    --agent.num_qs=10 \
    --agent.rho=0.0 \
    --agent.batch_size=256 \
    --agent.score_mode=${SCORE_MODE} \
    --agent.score_net_hidden_dims=${CURRENT_DIMS} \
    --agent.score_sigma_min=1e-4 \
    --offline_steps=1000000 \
    --online_steps=500000 \
    --eval_interval=50000 \
    --save_interval=500000 \
    --dataset_replace_interval=2000000 \
    --save_dir="/models/saved_models/job_${JOB_INDEX}_humanoidmaze_large_task${TASK_ID}_tauC${TAU_CRITIC}_tauS${TAU_SCORE}_mix${MIXTURE_PROB}"
