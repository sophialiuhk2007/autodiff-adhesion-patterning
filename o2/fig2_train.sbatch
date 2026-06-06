#!/usr/bin/env bash
#SBATCH --job-name=jxm_fig2_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
## If O2 requires an account for your GPU partition, uncomment and edit:
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

# Load cluster modules if your O2 environment needs them. Adjust CUDA to whatever
# `module avail cuda` shows on O2; JAX from environment.yml expects CUDA 12.
if command -v module >/dev/null 2>&1; then
    module purge
    module load gcc/9.2.0 || true
    module load cuda/12.4 || module load cuda/12.2 || module load cuda/12.1 || true
fi

# Activate the environment you create on O2.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi
conda activate jax-morph

python - <<'PY'
import jax
print("JAX devices:", jax.devices())
PY

cd results-natcompsci-2025/figure_2
python fig2_train.py

echo "End: $(date)"
