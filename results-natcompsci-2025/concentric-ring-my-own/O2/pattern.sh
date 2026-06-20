#!/bin/bash
#SBATCH -J salt_pepper_patterns
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=150G
#SBATCH -o logs/salt_pepper_patterns_%j.out
#SBATCH -e logs/salt_pepper_patterns_%j.err

cd /home/sol856/jax-morph

module purge
module load gcc/14.2.0
module load python/3.13.1

source .venv/bin/activate

mkdir -p logs

which python3
python3 --version

python3 results-natcompsci-2025/concentric-ring-my-own/train_logged_patterns.py \
  --pattern salt-pepper-shell \
  --n-opt-runs 5 \
  --epochs 200 \
  --n-episodes 10 \
  --n-steps 20 \
  --n-cells 120 \
  --w12 1 \
  --w33 1 \
  --w-media1 1 \
  --w-media2 1 \
  --w-media3 1 \
  --w-frac 0 \
  --learning-rate 0.1 \
  --final-learning-rate 0.001 \
  --lr-schedule exponential \
  --outdir ./trained_models_patterns_salt_pepper \
  --clean
