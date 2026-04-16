#!/bin/bash
#BSUB -J train_staircase_r1
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_staircase_r1_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_staircase_r1_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# Staircase step 1: fine-tune feb_sf on merged distributions.
#
# Hypothesis: geometry expands in discrete jumps when qualitatively new signal
# is introduced. feb_sf (rank 18.9) opened the draw dimension via min_decisive=0.0.
# This run adds distributional breadth (January balanced positions) on top of the
# existing geometry without retraining from scratch.
#
# Key choices:
#   --init-model    start from feb_sf geometry (rank 18.9), don't discard it
#   --lr 3e-4       lower than default 1e-3 — preserve existing dimensions,
#                   let new signal expand rather than overwrite
#   --extra-dataset dataset_balanced.pt adds January full-board balanced positions
#                   (~200k) — different month, different distribution, more variety
#   --anchor-dataset holds win/loss endgame axis stable while middle expands
#   dataset_sf.pt skipped — decisive endgame signal is covered by the anchor
#
# Total: ~600k primary+extra positions + anchor fraction.
#
# Gate (run probe_geometry.sh after training):
#   Effective rank > 22 — if staircase idea is right, rank should move
#   win·draw cosine (strict) closer to 0 — draw cluster should separate further

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset        /zhome/81/b/206091/Petra-ver2/data/dataset_feb_sf.pt \
    --extra-dataset  /zhome/81/b/206091/Petra-ver2/data/dataset_balanced.pt \
    --anchor-dataset /zhome/81/b/206091/Petra-ver2/data/endgame_anchor.pt \
    --anchor-frac    0.15 \
    --init-model     /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --lr             3e-4 \
    --rank-reg       0.1 \
    --weight-decay   5e-4 \
    --num-workers    4 \
    --out            /zhome/81/b/206091/Petra-ver2/models/staircase_r1
