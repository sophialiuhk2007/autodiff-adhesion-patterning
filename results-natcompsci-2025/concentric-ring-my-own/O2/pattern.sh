#!/bin/bash
#SBATCH -J bilobed_patterns
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=150G
#SBATCH -o logs/bilobed_patterns_%j.out
#SBATCH -e logs/bilobed_patterns_%j.err

cd /home/sol856/jax-morph

source .venv/bin/activate

mkdir -p logs

python results-natcompsci-2025/concentric-ring-my-own/train_logged_patterns.py \
  --pattern bilobed-shell \
  --n-opt-runs 5 \
  --epochs 200 \
  --n-episodes 10 \
  --n-steps 20 \
  --n-cells 120 \
  --type1-compactness-weight 5 \
  --type2-compactness-weight 5 \
  --w-centroid-distance 10 \
  --w11 1 \
  --w22 1 \
  --w12 1 \
  --w13 1 \
  --w23 1 \
  --w33 1 \
  --w-media1 5 \
  --w-media2 5 \
  --w-media3 5 \
  --w-frac 0 \
  --learning-rate 0.1 \
  --final-learning-rate 0.001 \
  --lr-schedule exponential \
  --outdir ./trained_models_patterns_bilobed_ring_contacts \
  --clean