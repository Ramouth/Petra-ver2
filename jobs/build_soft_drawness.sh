#!/bin/bash
set -euo pipefail
#BSUB -J build_soft_drawness
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/build_soft_drawness_%J.out
#BSUB -e /zhome/81/b/206091/logs/build_soft_drawness_%J.err

# Compute outcome-smoothed drawness targets for the curriculum.
# For each position: drawness_soft_target = fraction of k=50 nearest geometric
# neighbours whose game ended in a draw. Uses phase15_mid_no_endgame backbone
# for the geometry (rank ~76, stable).
#
# CPU-only: geometry extraction is small (500k forward passes of a 1.3M param
# net), k-NN is sklearn brute-force cosine — both run fine on hpc nodes.
# n=16 cores so sklearn n_jobs=-1 parallelises the k-NN distance computation.
#
#   bsub < jobs/build_soft_drawness.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_drawness_curriculum.pt"
MODEL="${HOME_DIR}/Petra-ver2/models/phase15_mid_no_endgame/best.pt"
OUT="${BLACKHOLE}/dataset_drawness_curriculum_soft.pt"

echo "=== Building soft drawness targets (CPU) ==="
echo "Dataset: ${DATASET}"
echo "Model:   ${MODEL}"
echo "Out:     ${OUT}"
echo

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/build_soft_drawness.py" \
    --dataset "${DATASET}" \
    --model   "${MODEL}" \
    --out     "${OUT}" \
    --k       50

echo
echo "Done. Soft-target dataset: ${OUT}"
