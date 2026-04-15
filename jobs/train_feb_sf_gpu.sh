#!/bin/bash
#BSUB -J train_feb_sf
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_feb_sf_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_feb_sf_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# Training on February 2020 SF-relabelled dataset.
#
# Goal: open the draw dimension.
#   dataset_feb_sf.pt uses min_decisive=0.0, so it covers the full decisiveness
#   spectrum including drawn and equal positions — the key difference from all
#   prior datasets that used min_decisive ≥ 0.05.
#
# Dataset: ~400k positions, full-board (max_pieces=32), depth-15 SF labels.
#   No --extra-dataset: single consistent distribution.
#   No --anchor-dataset: this run is intended to measure what the broad February
#     distribution does on its own. On a very large primary dataset, the current
#     anchor set would also be too diluted to provide a true 20% mix unless it
#     is oversampled or mixed per batch.
#
# Memory: train.py no longer pre-materialises visit_dists or legal_masks for
#   datasets without stored visit_dists — one-hot and mask unpacking happen
#   per batch. Estimated peak host RAM: ~6-8 GB.
#
# Geometry gate (run probe_geometry.sh after training):
#   - win·draw centroid cosine closer to 0 than previous runs (0.2657)
#   - Effective rank ≥ 10 (draw dimension should separate from win/loss axis)
#
# ELO gate (run eval_round2_vs_material.sh):
#   - Greedy vs Random > 70%
#   - MCTS(learned) vs MCTS(material) > 55%

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset      /zhome/81/b/206091/Petra-ver2/data/dataset_feb_sf.pt \
    --rank-reg     0.1 \
    --num-workers  4 \
    --weight-decay 5e-4 \
    --out          /zhome/81/b/206091/Petra-ver2/models/feb_sf
