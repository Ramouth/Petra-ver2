#!/bin/bash
set -euo pipefail
#BSUB -J probe_geometry_drawness_full_natural_init
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_geometry_drawness_full_natural_init_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_geometry_drawness_full_natural_init_%J.err

# Geometry probe on drawness_full_natural_init (EXP 2 result).
#
# What to compare against:
#   natural:                rank 38.7  w·l cos -0.65  LR 0.798  PC1-d 1.43  β1 192
#   drawness_full:          rank 34.2  (probe pending in EXP 3)
#   big_2025_filtered:      rank 28.9  w·l cos +0.03 LR 0.738  PC1-d 1.24  β1 124
#
# EXP 2 prediction:
#   - rank ≥ 30 (held by both natural and drawness_full)
#   - w·l cosine negative (the BCE+draw_reg result preserved)
#   - drawness curriculum kept the topology alive (β1 high)
#
#   bsub < jobs/probe_geometry_drawness_full_natural_init.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODEL="${HOME_DIR}/Petra-ver2/models/drawness_full_natural_init/best.pt"
DATASET="/dtu/blackhole/0b/206091/dataset_drawness_curriculum.pt"

if [ ! -f "${MODEL}" ];   then echo "ERROR: not found: ${MODEL}";   exit 1; fi
if [ ! -f "${DATASET}" ]; then echo "ERROR: not found: ${DATASET}"; exit 1; fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/probe_geometry.py" \
    --model   "${MODEL}" \
    --dataset "${DATASET}" \
    --n       5000

echo "Done."
