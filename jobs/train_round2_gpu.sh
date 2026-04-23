#!/bin/bash
#BSUB -J train_round2
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_round2_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_round2_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# Round 2 supervised training — full pieces, mixed decisiveness.
#
# Goal: push effective rank from 7.2 → 30+, open draw dimension.
#
# Changes vs sf_balanced (train_balanced_gpu.sh):
#   - Primary dataset: dataset_balanced.pt (max_pieces=32, min_decisive=0.05)
#     This is full-board only — no max_pieces=20 restriction.
#     sf_balanced used dataset_sf.pt (max_pieces=20) as primary; that caused
#     the starting position to be OOD (→ labelled "loss" in the probe).
#   - No extra_dataset merge: training on one consistent full-board distribution.
#   - --rank-reg 0.1: adds λ·tr(C²) to loss each batch.
#     tr(C²) = 1/eff_rank for unit vectors. At rank=7.2 this adds ~0.014 to
#     loss — small but provides a direct gradient toward rank expansion.
#     The covariance penalty also pushes the draw cluster off the win/loss axis,
#     naturally opening a draw dimension.
#   - --anchor-frac 0.20: slightly more endgame anchor (up from 0.15) to keep
#     the win/loss axis oriented as rank expands into new dimensions.
#   - Cold start (no --init-model): sf_balanced's 7D subspace may be a local
#     attractor. Starting fresh lets the rank reg shape the geometry from scratch.
#
# Geometry gate after training (run probe_geometry.sh):
#   - Effective rank > 15 (stretch: > 30)
#   - win·draw centroid cosine should move toward 0 (draw dimension opening)
#   - All sanity checks must still pass
#
# ELO gate (run eval_round2_vs_material.sh):
#   - Step 2: Greedy vs Random > 70% (policy must survive full-board training)
#   - Step 5: MCTS(learned) vs MCTS(material) > 55%

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset        /zhome/81/b/206091/Petra-ver2/data/dataset_balanced.pt \
    --anchor-dataset /zhome/81/b/206091/Petra-ver2/data/endgame_anchor.pt \
    --anchor-frac    0.12 \
    --rank-reg       0.1 \
    --weight-decay   5e-4 \
    --out            /zhome/81/b/206091/Petra-ver2/models/round2
