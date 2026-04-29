#!/bin/bash
set -euo pipefail
#BSUB -J train_natural_v3
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=64GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_natural_v3_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_natural_v3_%J.err

# Natural v3: same recipe as natural, much bigger data (4.5M @ 7.5% drawn)
# pooled from all high-quality SF-reevaled sources.
#
# Hypothesis: scale wins. With 10x more data than natural and the same
# recipe (no drawness scaffolding), geometry should organise drawness
# more cleanly while preserving more rank.
#
# Patience 10 — bigger data, longer reshape expected.
# Memory 64GB — 4.5M positions × ~1KB/row ≈ 4-5GB tensor + overhead.
#
#   bsub < jobs/build_natural_v3.sh   # build the dataset first
#   bsub < jobs/train_natural_v3.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_natural_v3.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/2021_06_all/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/natural_v3"

echo "=== Natural v3 training (4.5M @ 7.5% drawn, no scaffolding) ==="
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
echo "Next: bsub -env \"MODEL_DIR=natural_v3\" < jobs/probe_drawness_head.sh"
