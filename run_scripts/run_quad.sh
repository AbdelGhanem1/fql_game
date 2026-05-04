#!/bin/bash

# Usage: ./run_quadruple.sh [JOB_INDEX]
JOB_INDEX=${1:-0}

# ==============================================================================
# 1. PARAMETER SELECTION
# ==============================================================================
SEEDS=(40004 10001 20002 50005)
ALPHAS=(0.5)

TEMPS=(0.5)
TASKS=(4 1)
SCORE_MODES=("fast")
# Quoted strings to preserve brackets/commas
HIDDEN_DIMS=("[512,512,512,512]")

MIXTURES=(0.9)
# NEW PARAMETERS
TAU_CRITICS=(0.1)
TAU_SCORES=(0.1)

NUM_SEEDS=${#SEEDS[@]}
NUM_ALPHAS=${#ALPHAS[@]}
NUM_MIXTURES=${#MIXTURES[@]}
NUM_TEMPS=${#TEMPS[@]}
NUM_TASKS=${#TASKS[@]}
NUM_MODES=${#SCORE_MODES[@]}
NUM_DIMS=${#HIDDEN_DIMS[@]}
NUM_TAU_CRITICS=${#TAU_CRITICS[@]}
NUM_TAU_SCORES=${#TAU_SCORES[@]}

# ==============================================================================
# Indexing Logic (Ordered fastest-changing to slowest-changing)
# ==============================================================================

# 1. TASKS
TASK_ID=${TASKS[$(( JOB_INDEX % NUM_TASKS ))]}

# 2. SEEDS
SEED=${SEEDS[$(( (JOB_INDEX / NUM_TASKS) % NUM_SEEDS ))]}

# 3. TEMPS (Mapped correctly to inverse temperature logic)
INV_TEMP=${TEMPS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS)) % NUM_TEMPS ))]}

# 4. ALPHAS
ME_AM_ALPHA=${ALPHAS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS)) % NUM_ALPHAS ))]}

# 5. MIXTURES
MIXTURE_PROB=${MIXTURES[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS * NUM_ALPHAS)) % NUM_MIXTURES ))]}

# 6. SCORE_MODES
SCORE_MODE=${SCORE_MODES[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS * NUM_ALPHAS * NUM_MIXTURES)) % NUM_MODES ))]}

# 7. HIDDEN_DIMS
CURRENT_DIMS=${HIDDEN_DIMS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS * NUM_ALPHAS * NUM_MIXTURES * NUM_MODES)) % NUM_DIMS ))]}

# 8. TAU_CRITICS
TAU_CRITIC=${TAU_CRITICS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS * NUM_ALPHAS * NUM_MIXTURES * NUM_MODES * NUM_DIMS)) % NUM_TAU_CRITICS ))]}

# 9. TAU_SCORES 
TAU_SCORE=${TAU_SCORES[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_TEMPS * NUM_ALPHAS * NUM_MIXTURES * NUM_MODES * NUM_DIMS * NUM_TAU_CRITICS)) % NUM_TAU_SCORES ))]}


# Create a "clean" version of dims for filenames
DIMS_TAG=$(echo $CURRENT_DIMS | tr -d '[],')

echo "=========================================="
echo "Workstation Job Index: $JOB_INDEX"
echo "Config: Task=$TASK_ID | Mode=$SCORE_MODE | Dims=$CURRENT_DIMS | Alpha=$ME_AM_ALPHA | TauC=$TAU_CRITIC | TauS=$TAU_SCORE"
echo "=========================================="

# ==============================================================================
# 2. ENVIRONMENT SETUP
# ==============================================================================
CONDA_ENV="$HOME/micromamba/envs/fql_env"
SITE_PACKAGES="$CONDA_ENV/lib/python3.10/site-packages"

# CRITICAL FIX: Define the absolute path to Python
PYTHON_EXEC="$CONDA_ENV/bin/python"

# Standard Library Paths
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:$LD_LIBRARY_PATH"

# Point XLA to the CUDA compiler
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$SITE_PACKAGES/nvidia/cuda_runtime/../.. --xla_gpu_strict_conv_algorithm_picker=false"

# Rendering & Memory Prefs
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
# Updated to the requested cube-quadruple directory name
DATASET_DIR="$HOME/abdelghani_work/datasets/cube-quadruple-play-100m"

cd "$PROJECT_DIR"
mkdir -p logs saved_models

# Check for dataset files (Updated Renaming logic for cube-quadruple)
if ls "$DATASET_DIR"/cube-quadruple-play*000.npz 1> /dev/null 2>&1; then
    echo "⚠️  Found mismatching filenames. Auto-renaming..."
    cd "$DATASET_DIR"
    for f in cube-quadruple-play*.npz; do
        [ -e "$f" ] || continue
        num=$(echo "$f" | grep -oP '\d+(?=\.npz|-val\.npz)' | sed 's/^0*//')
        [ -z "$num" ] && num=0
        if [[ "$f" == *"val"* ]]; then 
            mv "$f" "shard_${num}-val.npz"
        else 
            mv "$f" "shard_${num}.npz"
        fi
    done
    cd "$PROJECT_DIR"
    echo "✅ Dataset renamed."
fi

# WandB Config - Updated Project and Run Group Naming
export WANDB_PROJECT="meam-cube-quadruple_v25_sampling"
export WANDB_NAME="task${TASK_ID}_tmp${INV_TEMP}_mprob${MIXTURE_PROB}_${SCORE_MODE}_dims${DIMS_TAG}_alpha${ME_AM_ALPHA}_tauC${TAU_CRITIC}_tauS${TAU_SCORE}_seed${SEED}"

echo "🚀 Starting Training on Cube-Quadruple..."

rm -f /dev/shm/meam_worker_*.npz

# CRITICAL FIX: Use $PYTHON_EXEC instead of just 'python'
"$PYTHON_EXEC" main.py \
    --run_group=cube-quadruple_Workstation_Repro \
    --agent=agents/meam.py \
    --seed=${SEED} \
    --env_name=cube-quadruple-play-singletask-task${TASK_ID}-v0 \
    --ogbench_dataset_dir="${DATASET_DIR}" \
    --sparse=False \
    --horizon_length=5 \
    --agent.action_chunking=True \
    --agent.inv_temp=${INV_TEMP} \
    --agent.mixture_prob=${MIXTURE_PROB} \
    --agent.me_am_alpha=${ME_AM_ALPHA} \
    --agent.tau_critic=${TAU_CRITIC} \
    --agent.tau_score=${TAU_SCORE} \
    --agent.num_qs=10 \
    --agent.rho=0.5 \
    --agent.score_sigma_min=3e-4\
    --agent.batch_size=256 \
    --agent.score_mode=${SCORE_MODE} \
    --agent.score_net_hidden_dims=${CURRENT_DIMS} \
    --offline_steps=1000000 \
    --online_steps=100000 \
    --eval_interval=50000 \
    --save_interval=500000 \
    --save_dir="./saved_models/cube_quad_job_${JOB_INDEX}_task${TASK_ID}_dims${DIMS_TAG}_${SCORE_MODE}_tauC${TAU_CRITIC}_tauS${TAU_SCORE}"
