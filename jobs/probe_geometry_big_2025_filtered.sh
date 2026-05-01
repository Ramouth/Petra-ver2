#!/bin/bash
set -euo pipefail
#BSUB -J probe_geo_big_2025_filtered
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_geo_big_2025_filtered_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_geo_big_2025_filtered_%J.err

# Geometry probe on big_2025_filtered checkpoints.
# Compare best.pt (epoch 1, rank 25.4, wdcos 0.6651) vs epoch_09.pt
# (rank 19.2, wdcos 0.2357 — best draw geometry of the run).
# Probes: eigenvalue distribution, label separation, known FENs (KR vs KR
# etc.), drawness head, nearest neighbours, equal-subspace.
#
#   bsub < jobs/probe_geometry_big_2025_filtered.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODELS="${HOME_DIR}/Petra-ver2/models/big_2025_filtered"
DATASET="/dtu/blackhole/0b/206091/dataset_2025_filtered_pc23.pt"

if [ ! -f "${DATASET}" ]; then
    echo "ERROR: dataset not found: ${DATASET}"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

for CKPT in best.pt epoch_09.pt; do
    MODEL="${MODELS}/${CKPT}"
    if [ ! -f "${MODEL}" ]; then
        echo "WARNING: skipping ${CKPT} — not found at ${MODEL}"
        continue
    fi
    echo
    echo "######################################################################"
    echo "##  big_2025_filtered/${CKPT}"
    echo "######################################################################"
    python3 -u "${SRC}/probe_geometry.py" \
        --model   "${MODEL}" \
        --dataset "${DATASET}" \
        --n       5000
done

echo
echo "Done."
