#!/bin/bash
#BSUB -J eval_feb_sf_vs_material
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_round2_vs_material_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_round2_vs_material_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# ELO evaluation: feb_sf model vs MCTS(material).
# Baselines (round2 / sf_balanced):
#   Step 2 (greedy vs random):        sf_balanced 75%
#   Step 5 (MCTS learned vs material): sf_balanced 58% (ELO +56)
#
# Gates for feb_sf:
#   Step 2 > 70% — policy must survive full-board, draw-inclusive training
#   Step 5 > 55% — value must beat material heuristic to justify self-play

python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model   /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --step    2 \
    --games   100 \
    --workers 16

python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model   /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --step    5 \
    --n-sim   100 \
    --games   200 \
    --workers 16
