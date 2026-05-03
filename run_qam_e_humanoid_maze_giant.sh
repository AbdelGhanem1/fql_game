#!/bin/bash

# Usage: ./run_qam_e_humanoid_runpod.sh [JOB_INDEX]
JOB_INDEX=${1:-0}

# ==============================================================================
# CONFIGURATION ARRAYS
# ==============================================================================
SEEDS=(10001 20002 30003 40004 50005 60006 70007 80008)
TASKS=(1 2 3)

INV_TEMPS=(5.0)     
DISCOUNTS=(0.995)   

NUM_SEEDS=${#SEEDS[@]}
NUM_TASKS=${#TASKS[@]}
NUM_INV_TEMPS=${#INV_TEMPS[@]}
NUM_DISCOUNTS=${#DISCOUNTS[@]}

# ==============================================================================
# PARAMETER SELECTION
# ==============================================================================
TASK_ID=${TASKS[$(( JOB_INDEX % NUM_TASKS ))]}
SEED=${SEEDS[$(( (JOB_INDEX / NUM_TASKS) % NUM_SEEDS ))]}
INV_TEMP=${INV_TEMPS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS)) % NUM_INV_TEMPS ))]}
DISCOUNT=${DISCOUNTS[$(( (JOB_INDEX / (NUM_TASKS * NUM_SEEDS * NUM_INV_TEMPS)) % NUM_DISCOUNTS ))]}

echo "=========================================="
echo "RunPod Job Index: $JOB_INDEX"
echo "Config: Humanoidmaze-Large Task=$TASK_ID | Seed=$SEED"
echo "Params: QAM-E (Edit) | Gamma=$DISCOUNT | InvTemp=$INV_TEMP"
echo "=========================================="

# ==============================================================================
# RUNPOD DATASET OPTIMIZATION (RAM DISK)
# ==============================================================================
PERSISTENT_DATASET_DIR="/workspace/datasets/humanoidmaze-large"
RAMDISK_DATASET_DIR="/dev/shm/humanoidmaze-large"

# Copy to RAM disk if it's not already there
if [ ! -d "$RAMDISK_DATASET_DIR" ]; then
    echo "⚡ Copying dataset to high-speed RAM disk (/dev/shm)..."
    mkdir -p /dev/shm
    cp -r "$PERSISTENT_DATASET_DIR" /dev/shm/
fi

# ==============================================================================
# ENVIRONMENT SETUP (RUNPOD DOCKER)
# ==============================================================================
export NVIDIA_TF32_OVERRIDE=1
export JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export JAX_ENABLE_X64=False
export PYTHONUNBUFFERED=1

# RunPod Persistent Output Directories
cd /workspace/fql_game
mkdir -p /workspace/logs /workspace/saved_models

export WANDB_PROJECT="humanoidmaze-large_qam_E_baseline"
export WANDB_NAME="runpod_qam_e_task${TASK_ID}_invTemp${INV_TEMP}_seed${SEED}"

echo "🚀 Starting QAM-E Training on RunPod..."

ENV_NAME="humanoidmaze-large-navigate-singletask-task${TASK_ID}-v0"

python main.py \
    --run_group=humanoidmaze-large_RunPod_QAM_E \
    --agent=agents/qam.py \
    --tags=RUNPOD,QAM_E,task${TASK_ID},SEED_${SEED} \
    --seed=${SEED} \
    --env_name=${ENV_NAME} \
    --ogbench_dataset_dir="${RAMDISK_DATASET_DIR}" \
    --sparse=False \
    --horizon_length=5 \
    --agent.action_chunking=False \
    --balanced_sampling=False \
    --agent.discount=${DISCOUNT} \
    --agent.inv_temp=${INV_TEMP} \
    --agent.flow_steps=10 \
    --agent.fql_alpha=0.0 \
    --agent.edit_scale=0.1 \
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
    --save_dir="/workspace/saved_models/job_${JOB_INDEX}_qam_e_humanoidmaze_large_task${TASK_ID}_invTemp${INV_TEMP}"

echo "✅ Job ${JOB_INDEX} complete for Task ${TASK_ID}!"
