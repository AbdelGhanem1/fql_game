#!/usr/bin/env bash
set -euo pipefail

# Run one job:
#   ./run_meam_geo_antmaze_giant.sh JOB_INDEX
#
# Sweep layout:
#   5 tasks x 3 seeds x 4 mixture probabilities = 60 jobs
#   valid JOB_INDEX values: 0 ... 59
#
# mixture=0.0 is the in-code vanilla QAM control.
# mixture>0.0 enables only the Gaussian geometric expansion module.

JOB_INDEX="${1:-0}"

SEEDS=(40004 10001 20002)
TASKS=(1 2 3 4 5)
MIXTURES=(0.0 0.1 0.2 0.3)

NUM_SEEDS=${#SEEDS[@]}
NUM_TASKS=${#TASKS[@]}
NUM_MIXTURES=${#MIXTURES[@]}
TOTAL_JOBS=$((NUM_TASKS * NUM_SEEDS * NUM_MIXTURES))

if ! [[ "${JOB_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: JOB_INDEX must be a non-negative integer." >&2
    exit 2
fi

if (( JOB_INDEX < 0 || JOB_INDEX >= TOTAL_JOBS )); then
    echo "ERROR: JOB_INDEX=${JOB_INDEX} is out of range." >&2
    echo "Valid range: 0-$((TOTAL_JOBS - 1))" >&2
    exit 2
fi

# Tasks vary fastest, then seeds, then mixture probability.
TASK_ID=${TASKS[$((JOB_INDEX % NUM_TASKS))]}
SEED=${SEEDS[$(((JOB_INDEX / NUM_TASKS) % NUM_SEEDS))]}
MIXTURE_PROB=${MIXTURES[$(((JOB_INDEX / (NUM_TASKS * NUM_SEEDS)) % NUM_MIXTURES))]}

# Fixed QAM/AntMaze-Giant settings.
INV_TEMP=3.0
DISCOUNT=0.995
TAU_CRITIC_BACKUP=0.5
GEO_ACTION_SCALE=1.0
GEO_ENTROPY_MULTIPLIER=0.5

echo "============================================================"
echo "MEAM_geo: QAM + training-time Gaussian geometric expansion"
echo "Job       : ${JOB_INDEX}/${TOTAL_JOBS}"
echo "Task      : antmaze-giant task ${TASK_ID}"
echo "Seed      : ${SEED}"
echo "Mixture   : ${MIXTURE_PROB}"
echo "QAM beta^-1: ${INV_TEMP}"
echo "Discount  : ${DISCOUNT}"
echo "Geo entropy target: -${GEO_ENTROPY_MULTIPLIER} * action_dim"
echo "============================================================"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-$HOME/micromamba/envs/fql_env}"
PYTHON_EXEC="${CONDA_ENV}/bin/python"
SITE_PACKAGES="${CONDA_ENV}/lib/python3.10/site-packages"

if [[ ! -x "${PYTHON_EXEC}" ]]; then
    echo "ERROR: Python executable not found: ${PYTHON_EXEC}" >&2
    exit 3
fi

export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${SITE_PACKAGES}/nvidia/nvjitlink/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${SITE_PACKAGES}/nvidia/cuda_runtime/../.. --xla_gpu_strict_conv_algorithm_picker=false"

export NVIDIA_TF32_OVERRIDE=1
export JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_ENABLE_X64=False
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/abdelghani_work/datasets/antmaze-giant}"

cd "${PROJECT_DIR}"
mkdir -p logs saved_models

export WANDB_PROJECT="${WANDB_PROJECT:-meam_geo_antmaze_giant}"
export WANDB_NAME="geo_task${TASK_ID}_mix${MIXTURE_PROB}_temp${INV_TEMP}_gamma${DISCOUNT}_seed${SEED}"

SAVE_DIR="./saved_models/meam_geo/job_${JOB_INDEX}_task${TASK_ID}_mix${MIXTURE_PROB}_seed${SEED}"

echo "Dataset dir: ${DATASET_DIR}"
echo "Save dir   : ${SAVE_DIR}"
echo "Starting training..."

"${PYTHON_EXEC}" main.py \
    --run_group=antmaze-giant_MEAM_geo \
    --agent=agents/meam_geo.py \
    --seed="${SEED}" \
    --env_name="antmaze-giant-navigate-singletask-task${TASK_ID}-v0" \
    --ogbench_dataset_dir="${DATASET_DIR}" \
    --sparse=False \
    --horizon_length=1 \
    --agent.action_chunking=False \
    --balanced_sampling=False \
    --agent.discount="${DISCOUNT}" \
    --agent.inv_temp="${INV_TEMP}" \
    --agent.num_qs=10 \
    --agent.rho="${TAU_CRITIC_BACKUP}" \
    --agent.batch_size=256 \
    --agent.best_of_n=1 \
    --agent.residual=False \
    --agent.target_actor=True \
    --agent.use_target_grad=True \
    --agent.clip_adj=True \
    --agent.clip_grad=True \
    --agent.geo_mixture_prob="${MIXTURE_PROB}" \
    --agent.geo_action_scale="${GEO_ACTION_SCALE}" \
    --agent.geo_use_mode_for_targets=True \
    --agent.geo_use_entropy=True \
    --agent.geo_target_entropy_multiplier="${GEO_ENTROPY_MULTIPLIER}" \
    --offline_steps=1000000 \
    --online_steps=0 \
    --eval_interval=50000 \
    --save_interval=500000 \
    --dataset_replace_interval=2000000 \
    --save_dir="${SAVE_DIR}"
