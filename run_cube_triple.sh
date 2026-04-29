#!/bin/bash

# Usage: ./run_cube_proxy.sh [JOB_INDEX]
JOB_INDEX=${1:-0}

# ==============================================================================
# 1. PARAMETER SELECTION FOR CUBE-TRIPLE PROXY TUNING
# ==============================================================================
# Tasks 2 and 4 represent the domain difficulty (Proxy tuning method)
TASKS=(2 5)
SEEDS=(10001 20002 30003)

# --- THE PAIRED COMBINATIONS (Alpha + Temp = 1.0) ---
# We will use the same index to access both arrays so they stay locked together.
# Pair 0: Alpha=0.1, Temp=0.9
# Pair 1: Alpha=0.3, Temp=0.7
ALPHAS=(0.0)
TEMPS=(1.0)

MIXTURES=(0.1)


TAU_CRITICS=(15.0)

# Slightly elevated from puzzle to encourage manifold exploration
TAU_SCORES=(1.0)

SCORE_MODES=("fast")
HIDDEN_DIMS=("[512,512,512,512]")

NUM_TASKS=${#TASKS[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_PAIRS=${#ALPHAS[@]} # This is 2. Controls both ALPHAS and TEMPS.
NUM_MIXTURES=${#MIXTURES[@]}
NUM_TAU_CRITICS=${#TAU_CRITICS[@]}
NUM_TAU_SCORES=${#TAU_SCORES[@]}
NUM_MODES=${#SCORE_MODES[@]}
NUM_DIMS=${#HIDDEN_DIMS[@]}

# ==============================================================================
# Indexing Logic (Zipping Alpha and Temp together)
# ==============================================================================

# 1. TASKS
TASK_ID=${TASKS[$(( JOB_INDEX % NUM_TASKS ))]}

# 2. SEEDS
SEED=${SEEDS[$(( (JOB_INDEX / NUM_TASKS) % NUM_SEEDS ))]}

# 3. THE ALPHA/TEMP PAIR
PAIR_IDX=$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS)) % NUM_PAIRS ))
ME_AM_ALPHA=${ALPHAS[$PAIR_IDX]}
INV_TEMP=${TEMPS[$PAIR_IDX]}

# 4. MIXTURES
MIXTURE_PROB=${MIXTURES[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_PAIRS)) % NUM_MIXTURES ))]}

# 5. TAU_CRITICS
TAU_CRITIC=${TAU_CRITICS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_PAIRS * NUM_MIXTURES)) % NUM_TAU_CRITICS ))]}

# 6. TAU_SCORES
TAU_SCORE=${TAU_SCORES[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_PAIRS * NUM_MIXTURES * NUM_TAU_CRITICS)) % NUM_TAU_SCORES ))]}

# Constants
SCORE_MODE=${SCORE_MODES[0]}
CURRENT_DIMS=${HIDDEN_DIMS[0]}

DIMS_TAG=$(echo $CURRENT_DIMS | tr -d '[],')

echo "=========================================="
echo "Workstation Job Index: $JOB_INDEX"
echo "Config: Task=$TASK_ID | Seed=$SEED | Mode=$SCORE_MODE"
echo "Convex Pair: Alpha=$ME_AM_ALPHA, Temp=$INV_TEMP"
echo "ME-AM Tuning: Mix=$MIXTURE_PROB | TauC=$TAU_CRITIC | TauS=$TAU_SCORE"
echo "=========================================="

# ==============================================================================
# 2. ENVIRONMENT SETUP
# ==============================================================================

CONDA_ENV="$HOME/micromamba/envs/fql_env"
SITE_PACKAGES="$CONDA_ENV/lib/python3.10/site-packages"
PYTHON_EXEC="$CONDA_ENV/bin/python"

export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$SITE_PACKAGES/nvidia/cuda_runtime/../.. --xla_gpu_strict_conv_algorithm_picker=false"

export NVIDIA_TF32_OVERRIDE=1
export JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export JAX_ENABLE_X64=False
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ==============================================================================
# 3. PATHS & TRAINING
# ==============================================================================
PROJECT_DIR="$(pwd)"
DATASET_DIR="$HOME/abdelghani_work/datasets/cube-triple" # UPDATE THIS IF NEEDED

cd "$PROJECT_DIR"
mkdir -p logs saved_models

# WandB Config - Renamed for Cube-Triple proxy tuning
export WANDB_PROJECT="cube_tripple_v2"
export WANDB_NAME="task${TASK_ID}_tmp${INV_TEMP}_mprob${MIXTURE_PROB}_${SCORE_MODE}_dims${DIMS_TAG}_alpha${ME_AM_ALPHA}_tauC${TAU_CRITIC}_tauS${TAU_SCORE}_seed${SEED}"

echo "🚀 Starting Training..."

rm -f /dev/shm/meam_worker_*.npz

"$PYTHON_EXEC" main.py \
    --run_group=cube-triple_Proxy_Tune \
    --agent=agents/meam.py \
    --seed=${SEED} \
    --env_name=cube-triple-play-singletask-task${TASK_ID}-v0 \
    --ogbench_dataset_dir="${DATASET_DIR}" \
    --sparse=False \
    --horizon_length=5 \
    --agent.action_chunking=True \
    --agent.inv_temp=${INV_TEMP} \
    --agent.clip_q_grad=False \
    --agent.mixture_prob=${MIXTURE_PROB} \
    --agent.me_am_alpha=${ME_AM_ALPHA} \
    --agent.tau_critic=${TAU_CRITIC} \
    --agent.tau_score=${TAU_SCORE} \
    --agent.num_qs=10 \
    --agent.rho=0.5 \
    --agent.batch_size=256 \
    --agent.score_mode=${SCORE_MODE} \
    --agent.score_net_hidden_dims=${CURRENT_DIMS} \
    --agent.use_gaussian_mode=True\
    --agent.score_sigma_min=3e-1\
    --offline_steps=1000000 \
    --online_steps=0 \
    --eval_interval=50000 \
    --save_interval=500000 \
    --dataset_replace_interval=2000000\
    --save_last_checkpoint=True\
    --save_dir="./saved_models/cube_job_${JOB_INDEX}_tk${TASK_ID}_a${ME_AM_ALPHA}_tC${TAU_CRITIC}_tS${TAU_SCORE}"
