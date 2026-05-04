#!/bin/bash

# Usage: ./run_qam_humanoid_workstation.sh [JOB_INDEX]
JOB_INDEX=${1:-0}

# ==============================================================================
# CONFIGURATION ARRAYS
# ==============================================================================
SEEDS=(50005 60006 70007 80008)
TASKS=(2 3 4)

INV_TEMPS=(5.0)     
DISCOUNTS=(0.995)   

NUM_SEEDS=${#SEEDS[@]}
NUM_TASKS=${#TASKS[@]}
NUM_INV_TEMPS=${#INV_TEMPS[@]}
NUM_DISCOUNTS=${#DISCOUNTS[@]}

# ==============================================================================
# PARAMETER SELECTION (Cascading Modulo)
# ==============================================================================
TASK_ID=${TASKS[$(( JOB_INDEX % NUM_TASKS ))]}
SEED=${SEEDS[$(( (JOB_INDEX / NUM_TASKS) % NUM_SEEDS ))]}
INV_TEMP=${INV_TEMPS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS)) % NUM_INV_TEMPS ))]}
DISCOUNT=${DISCOUNTS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_INV_TEMPS)) % NUM_DISCOUNTS ))]}

echo "=========================================="
echo "Workstation Job Index: $JOB_INDEX"
echo "Config: Humanoidmaze-Large Task=$TASK_ID | Seed=$SEED"
echo "Params: Pure QAM | Gamma=$DISCOUNT | InvTemp=$INV_TEMP"
echo "=========================================="

# ==============================================================================
# ENVIRONMENT SETUP (WORKSTATION OPTIMIZED)
# ==============================================================================
CONDA_ENV="$HOME/micromamba/envs/fql_env"
SITE_PACKAGES="$CONDA_ENV/lib/python3.10/site-packages"
PYTHON_EXEC="$CONDA_ENV/bin/python"

# Link CUDA/cuDNN locally
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:$LD_LIBRARY_PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$SITE_PACKAGES/nvidia/cuda_runtime/../.. --xla_gpu_strict_conv_algorithm_picker=false"

export NVIDIA_TF32_OVERRIDE=1
export JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export JAX_ENABLE_X64=False
export PYTHONUNBUFFERED=1

PROJECT_DIR="$(pwd)"
DATASET_DIR="$HOME/abdelghani_work/datasets/humanoidmaze-large"

cd "$PROJECT_DIR"
mkdir -p logs saved_models

export WANDB_PROJECT="humanoidmaze-large_qam_baseline"
export WANDB_NAME="ws_qam_task${TASK_ID}_invTemp${INV_TEMP}_seed${SEED}"

echo "🚀 Starting Pure QAM Training on Workstation..."

# Construct Env Name dynamically
ENV_NAME="humanoidmaze-large-navigate-singletask-task${TASK_ID}-v0"

"$PYTHON_EXEC" main.py \
    --run_group=humanoidmaze-large_Workstation_QAM \
    --agent=agents/qam.py \
    --tags=WORKSTATION,QAM,task${TASK_ID},SEED_${SEED} \
    --seed=${SEED} \
    --env_name=${ENV_NAME} \
    --ogbench_dataset_dir="${DATASET_DIR}" \
    --sparse=False \
    --horizon_length=5 \
    --agent.action_chunking=False \
    --balanced_sampling=False \
    --agent.discount=${DISCOUNT} \
    --agent.inv_temp=${INV_TEMP} \
    --agent.flow_steps=10 \
    --agent.fql_alpha=0.0 \
    --agent.edit_scale=0.0 \
    --agent.num_qs=10 \
    --agent.batch_size=256 \
    --agent.rho=0.0 \
    --offline_steps=1000000 \
    --online_steps=0 \
    --eval_episodes=50 \
    --log_interval=5000 \
    --eval_interval=50000 \
    --save_interval=500000 \
    --dataset_replace_interval=2000000 \
    --save_dir="./saved_models/job_${JOB_INDEX}_qam_humanoidmaze_large_task${TASK_ID}_invTemp${INV_TEMP}"

echo "✅ Job ${JOB_INDEX} complete for Task ${TASK_ID}!"
