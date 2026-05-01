#!/bin/bash
set -euo pipefail
#BSUB -J probe_geometry_natural
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_geometry_natural_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_geometry_natural_%J.err

# Geometry probe on natural + natural_v2 — the missing baseline.
# We've been comparing big_2025_filtered's equal-subspace LR (0.738, "WEAK")
# against nothing. This run gives us natural's number on the same probe.
#
# Probe outputs we care about:
#   - Effective rank (target ≥30)
#   - win·loss centroid cosine (target close to 0 or negative)
#   - win·draw cosine (the "wdcos" metric — natural was -0.20 in training)
#   - Equal-subspace LR (the most PoC-relevant — currently no baseline)
#   - β1 topology (target high)
#
#   bsub < jobs/probe_geometry_natural.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODELS="${HOME_DIR}/Petra-ver2/models"
DATASET="/dtu/blackhole/0b/206091/dataset_drawness_curriculum.pt"

if [ ! -f "${DATASET}" ]; then
    echo "ERROR: dataset not found: ${DATASET}"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

for MODEL_NAME in natural natural_v2; do
    MODEL="${MODELS}/${MODEL_NAME}/best.pt"
    if [ ! -f "${MODEL}" ]; then
        echo "WARNING: skipping ${MODEL_NAME} — not found at ${MODEL}"
        continue
    fi
    echo
    echo "######################################################################"
    echo "##  ${MODEL_NAME}"
    echo "######################################################################"
    python3 -u "${SRC}/probe_geometry.py" \
        --model   "${MODEL}" \
        --dataset "${DATASET}" \
        --n       5000
done

echo
echo "Done."
