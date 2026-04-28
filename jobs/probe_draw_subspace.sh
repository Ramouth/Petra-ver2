#!/bin/bash
#BSUB -J probe_draw_subspace
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_draw_subspace_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_draw_subspace_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Probe whether the equal-region geometry (|v|<0.2) already separates
# structural draws (KR vs KR, KNN vs K, KB vs KB) from balanced positions.
#
# Model : phase15_mid_no_endgame (rank 89.9) — latest competitive model
# Dataset: dataset_2021_06_mid_sf18.pt — fixed probe dataset (n=5000)
#
# Key output (Check 6):
#   accuracy > 0.85  → geometry already separates them
#                      frozen-backbone drawness head will work
#   accuracy 0.60-0.85 → weak separation, contrastive training may help
#   accuracy ~0.50   → no separation, backbone must change first

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"

python3 -u ${HOME_DIR}/Petra-ver2/src/probe_geometry.py \
    --model   ${HOME_DIR}/Petra-ver2/models/phase15_mid_no_endgame/best.pt \
    --dataset ${BLACKHOLE}/dataset_2021_06_mid_sf18.pt \
    --n 5000
