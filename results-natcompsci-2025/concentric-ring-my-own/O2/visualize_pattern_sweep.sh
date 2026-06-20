#!/bin/bash
#SBATCH -J sp_visualize
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=150G
#SBATCH -o logs/sp_visualize_%j.out
#SBATCH -e logs/sp_visualize_%j.err

set -euo pipefail

PROJECT_DIR=/home/sol856/jax-morph
SWEEP_ROOT="${PROJECT_DIR}/results-natcompsci-2025/concentric-ring-my-own/trained_models_patterns_salt_pepper_sweep"

cd "${PROJECT_DIR}"

module purge
module load gcc/14.2.0
module load python/3.13.1

source .venv/bin/activate

mkdir -p logs

export JAX_DISABLE_JIT=1
export JAX_TRACEBACK_FILTERING=off

which python3
python3 --version

python3 results-natcompsci-2025/concentric-ring-my-own/O2/visualize_pattern_sweep.py "${SWEEP_ROOT}"
