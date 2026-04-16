#!/bin/bash
#BSUB -J probe_staircase_r1
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_staircase_r1_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_staircase_r1_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Geometry probe: staircase_r1 vs feb_sf baseline.
# staircase_r1 = feb_sf fine-tuned on feb_sf + dataset_balanced.pt + anchor, lr=3e-4.
# Training stopped at epoch 1 (best) — model barely moved from feb_sf init.
# KR vs KR value regressed: +0.024 → +0.340 (draw dimension may have compressed).
#
# Expected: geometry unchanged or slightly worse vs feb_sf.
# feb_sf baseline:
#   Effective rank: 18.9
#   win·draw cosine (strict ±0.7/±0.3): 0.1676
#   win·loss cosine (strict):            -0.2465

python3 -u /zhome/81/b/206091/Petra-ver2/src/probe_geometry.py \
    --model   /zhome/81/b/206091/Petra-ver2/models/staircase_r1/best.pt \
    --dataset /zhome/81/b/206091/Petra-ver2/data/dataset_feb_sf.pt \
    --n 5000
