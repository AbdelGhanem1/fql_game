#!/bin/bash

# Usage: ./run_qam.sh [JOB_INDEX]
JOB_INDEX=${1:-0}

# ==============================================================================
# 1. PARAMETER SELECTION
# ==============================================================================
SEEDS=(40004 10001 20002 50005 80008 70007 60006 30003)
# Antmaze typically evaluates across tasks 1 through 5
TASKS=(2 5 3)

# --- QAM SPECIFIC ---
# In QAM, inv_temp acts equivalently to tau_critic in ME-AM.
# Taking your previous tau_critic value (5.0) for the antmaze-giant domain.
INV_TEMPS=(5.0)    

# --- DOMAIN SPECIFIC: antmaze-giant ---
DISCOUNTS=(0.995)   # CRITICAL: Long-horizon discount factor

NUM_SEEDS=${#SEEDS[@]}
NUM_TASKS=${#TASKS[@]}
NUM_INV_TEMPS=${#INV_TEMPS[@]}
NUM_DISCOUNTS=${#DISCOUNTS[@]}

# ==============================================================================
# Indexing Logic (Simplified for QAM)
# ==============================================================================
TASK_ID=${TASKS[$(( JOB_INDEX % NUM_TASKS ))]}
SEED=${SEEDS[$(( (JOB_INDEX / NUM_TASKS) % NUM_SEEDS ))]}

INV_TEMP=${INV_TEMPS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS)) % NUM_INV_TEMPS ))]}
DISCOUNT=${DISCOUNTS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_INV_TEMPS)) % NUM_DISCOUNTS ))]}

echo "=========================================="
echo "Workstation Job Index: $JOB_INDEX"
echo "Config: Antmaze-Giant Task=$TASK_ID | Seed=$SEED"
echo "Params: Gamma=$DISCOUNT | InvTemp=$INV_TEMP"
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
DATASET_DIR="$HOME/abdelghani_work/datasets/antmaze-giant"

cd "$PROJECT_DIR"
mkdir -p logs saved_models

export WANDB_PROJECT="vanilla_qam_antmaze_giant"
export WANDB_NAME="task${TASK_ID}_invTemp${INV_TEMP}_seed${SEED}"

echo "🚀 Starting QAM Training..."

"$PYTHON_EXEC" main.py \
    --run_group=antmaze-giant_Workstation_Repro_QAM \
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
    --agent.num_qs=10 \
    --agent.rho=0.5 \
    --agent.batch_size=256 \
    --offline_steps=1000000 \
    --online_steps=0 \
    --eval_interval=50000 \
    --save_interval=500000 \
    --dataset_replace_interval=2000000 \
    --save_dir="./saved_models/job_${JOB_INDEX}_qam_antmaze_giant_task${TASK_ID}_invTemp${INV_TEMP}"
