#!/usr/bin/env bash
#SBATCH --job-name=jxm_concentric_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
## If O2 requires an account for your partition, uncomment and edit:
## #SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Start: $(date)"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Keep matplotlib/JAX caches in writable project-local directories.
export MPLCONFIGDIR="${SLURM_SUBMIT_DIR}/.cache/matplotlib"
export XDG_CACHE_HOME="${SLURM_SUBMIT_DIR}/.cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

# Use the existing conda environment directly. This avoids relying on `conda`
# being initialized in Slurm's non-interactive shell.
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/jax-morph/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Could not find Python at ${PYTHON_BIN}"
    echo "Create the env first, or submit with PYTHON_BIN=/path/to/python sbatch o2/concentric_train.sbatch"
    exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import jax
print("JAX devices:", jax.devices())
PY

cd results-natcompsci-2025/concentric-ring-my-own

# By default, run the full paper-style training. For a quick O2 sanity test:
#   TRAIN_ARGS="--epochs 5 --n-opt-runs 1 --n-episodes 1 --n-val-episodes 1 --n-steps 2 --clean" sbatch o2/concentric_train.sbatch
# To save visual training previews, submit with:
#   TRAIN_SCRIPT=train_preview.py TRAIN_ARGS="--preview-interval 50 --clean" sbatch o2/concentric_train.sbatch
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"
TRAIN_ARGS="${TRAIN_ARGS:---clean}"
echo "Running: ${PYTHON_BIN} ${TRAIN_SCRIPT} ${TRAIN_ARGS}"
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" ${TRAIN_ARGS}

echo "End: $(date)"
