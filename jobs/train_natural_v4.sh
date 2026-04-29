#!/bin/bash
set -euo pipefail
#BSUB -J train_natural_v4
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=64GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_natural_v4_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_natural_v4_%J.err

# Natural v4: outcome-value supervision, 5M positions, 3 disjoint Lichess
# months. AlphaZero-style label signal — no SF eval as teacher, learns
# value purely from game outcomes.
#
# Same recipe (rank-reg, policy, no drawness scaffolding) as natural.
# Patience 10 — bigger data, longer reshape expected.
#
#   bsub < jobs/build_natural_v4.sh   # build the dataset first
#   bsub < jobs/train_natural_v4.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_natural_v4.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/2021_06_all/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/natural_v4"

echo "=== Natural v4 training (outcome supervision, 5M positions) ==="
echo "Dataset:    ${DATASET}"
echo "Init model: ${INIT_MODEL}"
echo "Out:        ${OUT_DIR}"
echo

for F in "${DATASET}" "${INIT_MODEL}"; do
    if [ ! -f "${F}" ]; then echo "ERROR: not found: ${F}"; exit 1; fi
done

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

python3 -u "${SRC}/train.py" \
    --dataset       "${DATASET}" \
    --init-model    "${INIT_MODEL}" \
    --out           "${OUT_DIR}" \
    --policy-weight 1.0 \
    --rank-reg      0.5 \
    --draw-reg      0.0 \
    --weight-decay  5e-4 \
    --lr            3e-4 \
    --batch-size    512 \
    --epochs        25 \
    --geo-patience  10

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
echo "Next: bsub -env \"MODEL_DIR=natural_v4\" < jobs/probe_drawness_head.sh"
