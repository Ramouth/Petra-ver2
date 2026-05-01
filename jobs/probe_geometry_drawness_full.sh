#!/bin/bash
set -euo pipefail
#BSUB -J probe_geometry_drawness_full
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_geometry_drawness_full_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_geometry_drawness_full_%J.err

# EXP 3 from the loop's Phase 3 plan: probe_geometry on drawness_full.
# Tells us what the geometry looks like when PoC AUC is 0.670 — the
# strongest geometry-level drawness signal in the project. Establishes the
# target signature for future training (esp. EXP 2's drawness_full_natural_init).
#
# Diagnostic output we care about:
#   - Equal-subspace LR  (natural=0.798, big_2025_f=0.738; what is drawness_full?)
#   - PC1 d on struct vs equal  (natural=1.43, natural_v2=1.94; expect drawness_full higher)
#   - Centroid cos struct vs equal  (lower = better separation)
#   - β1 topology (drawness_full's training showed 142 — confirm)
#   - Effective rank (training reported 34.2)
#
#   bsub < jobs/probe_geometry_drawness_full.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODEL="${HOME_DIR}/Petra-ver2/models/drawness_full/best.pt"
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
