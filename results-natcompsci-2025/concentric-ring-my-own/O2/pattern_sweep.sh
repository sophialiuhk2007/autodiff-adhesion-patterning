#!/bin/bash
#SBATCH -J sp_pattern_sweep
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=150G
#SBATCH --array=0-35
#SBATCH -o logs/sp_pattern_sweep_%A_%a.out
#SBATCH -e logs/sp_pattern_sweep_%A_%a.err

set -euo pipefail

PROJECT_DIR=/home/sol856/jax-morph
SCRIPT_DIR="${PROJECT_DIR}/results-natcompsci-2025/concentric-ring-my-own"

cd "${PROJECT_DIR}"

module purge
module load gcc/14.2.0
module load python/3.13.1

source .venv/bin/activate

mkdir -p logs

which python3
python3 --version

W_FRACS=(0 1 5)
W12S=(0 1)
TYPE_RATIO_LABELS=("1-1-8" "1-1-4")
TYPE_RATIO_VALUES=("1 1 8" "1 1 4")
N_CELLS_VALUES=(40 60 120)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

N_NCELLS=${#N_CELLS_VALUES[@]}
N_RATIOS=${#TYPE_RATIO_VALUES[@]}
N_W12S=${#W12S[@]}

NCELL_INDEX=$((TASK_ID % N_NCELLS))
RATIO_INDEX=$(((TASK_ID / N_NCELLS) % N_RATIOS))
W12_INDEX=$(((TASK_ID / (N_NCELLS * N_RATIOS)) % N_W12S))
WFRAC_INDEX=$((TASK_ID / (N_NCELLS * N_RATIOS * N_W12S)))

N_CELLS=${N_CELLS_VALUES[$NCELL_INDEX]}
TYPE_RATIO_LABEL=${TYPE_RATIO_LABELS[$RATIO_INDEX]}
TYPE_RATIO_STRING=${TYPE_RATIO_VALUES[$RATIO_INDEX]}
W12=${W12S[$W12_INDEX]}
W_FRAC=${W_FRACS[$WFRAC_INDEX]}

read -r TYPE_RATIO_1 TYPE_RATIO_2 TYPE_RATIO_3 <<< "$TYPE_RATIO_STRING"

OUTDIR="${SCRIPT_DIR}/trained_models_patterns_salt_pepper_sweep/wfrac_${W_FRAC}/w12_${W12}/ratio_${TYPE_RATIO_LABEL}/ncells_${N_CELLS}"

echo "started_at=$(date -Is)"
echo "slurm_job_id=${SLURM_JOB_ID:-local}"
echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-local}"
echo "slurm_array_task_id=${TASK_ID}"
echo "hostname=$(hostname)"
echo "cwd=$(pwd)"
echo "git_branch=$(git branch --show-current || true)"
echo "git_commit=$(git rev-parse HEAD || true)"
echo "git_status_short_start=$(git status --short || true)"
echo "config: w_frac=${W_FRAC}, w12=${W12}, type_ratios=${TYPE_RATIO_1} ${TYPE_RATIO_2} ${TYPE_RATIO_3}, n_cells=${N_CELLS}"
echo "outdir=${OUTDIR}"

python3 results-natcompsci-2025/concentric-ring-my-own/train_logged_patterns.py \
  --pattern salt-pepper-shell \
  --n-opt-runs 5 \
  --epochs 200 \
  --n-episodes 10 \
  --n-steps 20 \
  --n-cells "${N_CELLS}" \
  --type-ratios "${TYPE_RATIO_1}" "${TYPE_RATIO_2}" "${TYPE_RATIO_3}" \
  --w12 "${W12}" \
  --w33 1 \
  --w-media1 10 \
  --w-media2 10 \
  --w-media3 10 \
  --w-frac "${W_FRAC}" \
  --learning-rate 0.1 \
  --final-learning-rate 0.001 \
  --lr-schedule exponential \
  --outdir "${OUTDIR}" \
  --clean

python3 results-natcompsci-2025/concentric-ring-my-own/O2/summarize_pattern_sweep_run.py "${OUTDIR}"

echo "finished_at=$(date -Is)"
