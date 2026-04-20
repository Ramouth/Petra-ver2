#!/bin/bash
#BSUB -J train_2023_03
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_2023_03_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_2023_03_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# Training on Lichess 2023-03, 2500+ ELO, draw=0.0.
#
# Key changes vs feb_sf:
#   - draw label fixed to 0.0 (was -0.1) — draws now sit at the geometric
#     origin, cleanly separated from the loss cluster
#   - 2500+ ELO filter (vs 2200) — stronger players, more decisive positions
#   - ~2M positions from game outcomes (no SF relabelling)
#
# Geometry gate (run probe_geometry.sh after training):
#   - Effective rank > 20 (target: >30)
#   - win·draw cosine closer to 0 than feb_sf (0.1676)
#
# If rank plateaus near feb_sf levels despite clean data, the architecture
# (64ch × 4 blocks, shared policy/value bottleneck) is likely the ceiling.

BLACKHOLE="/dtu/blackhole/0b/206091"

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset      ${BLACKHOLE}/dataset_2023_03_sf18.pt \
    --rank-reg     0.1 \
    --num-workers  0 \
    --weight-decay 5e-4 \
    --out          /zhome/81/b/206091/Petra-ver2/models/lichess_2023_03
