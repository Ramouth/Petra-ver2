#!/bin/bash
#BSUB -J eval_round2_vs_material
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_round2_vs_material_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_round2_vs_material_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# ELO evaluation: round2 model vs MCTS(material).
# Baseline to beat: sf_balanced at 58% (ELO +56).
#
# Step 2: policy check — full-board training must not break the policy.
#   sf_balanced scored 75% here. Any drop signals the rank reg is hurting
#   policy quality and λ should be reduced.
#
# Step 5: value gate — must beat material at >55% to justify self-play.
#   This is the same threshold used throughout. sf_balanced was 58%.

python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model   /zhome/81/b/206091/Petra-ver2/models/round2/best.pt \
    --step    2 \
    --games   100 \
    --workers 16

python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model   /zhome/81/b/206091/Petra-ver2/models/round2/best.pt \
    --step    5 \
    --n-sim   100 \
    --games   200 \
    --workers 16
