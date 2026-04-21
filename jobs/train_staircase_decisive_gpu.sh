#!/bin/bash
#BSUB -J train_staircase_decisive
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_staircase_decisive_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_staircase_decisive_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# ── Staircase Stage A: decisive positions ──────────────────────────────────────
#
# Goal: push the win/loss axis into new positional dimensions, raising effective
# rank from the feb_sf baseline of 18.9 toward 30+.
#
# Data: decisive_*.pt — positions with |SF_value| ≥ 0.5, depth-20 labels,
#       high-ELO games (2200+).  These are unambiguously won or lost positions
#       where the network must learn structural (non-material) winning patterns.
#
# Init: feb_sf/best.pt — start from the current best geometry checkpoint.
#       lr=1e-4 (10× smaller than cold start) — fine-tuning, not retraining.
#
# Geometry gate (run probe_geometry.sh after training):
#   - Effective rank > 22 (at minimum; target > 30)
#   - win·loss cosine more negative than feb_sf (-0.2465)
#   - Top1 policy accuracy should hold (not regress by more than 5%)
#
# If rank does not improve from 18.9, the decisive dataset does not contain
# qualitatively new positional signal.  Check label std and position diversity.
#
# After this job: run train_staircase_balanced_gpu.sh (Stage B).

# ── Data paths — edit to match your split output ───────────────────────────────
# If you have multiple months, use --extra-dataset to merge them:
#   --dataset      data/decisive_2020_03.pt \
#   --extra-dataset data/decisive_2021_06.pt \

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset      /zhome/81/b/206091/Petra-ver2/data/decisive_2020_03.pt \
    --init-model   /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --out          /zhome/81/b/206091/Petra-ver2/models/staircase_decisive \
    --lr           1e-4 \
    --weight-decay 5e-4 \
    --rank-reg     0.1 \
    --epochs       20 \
    --geo-patience 3 \
    --num-workers  4
