#!/bin/bash
set -euo pipefail
#BSUB -J train_soft_drawness
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_soft_drawness_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_soft_drawness_%J.err

# Soft-drawness experiment (RL-flavored).
#
# Drawness target = empirical draw rate among k=50 geometric neighbours of
# each position. Replaces binary "is structurally drawn" with continuous
# "fraction of similar positions that drew" — empirical probability rather
# than static label.
#
# Companion to train_natural.sh:
#   natural        : no drawness scaffolding at all
#   soft_drawness  : drawness scaffolding with continuous outcome-derived target
#
#   bsub < jobs/build_soft_drawness.sh   # run first to build the dataset
#   bsub < jobs/train_soft_drawness.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_drawness_curriculum_soft.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/2021_06_all/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/soft_drawness"

echo "=== Soft-drawness training (outcome-smoothed targets) ==="
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
    --dataset                "${DATASET}" \
    --init-model             "${INIT_MODEL}" \
    --out                    "${OUT_DIR}" \
    --policy-weight          1.0 \
    --rank-reg               0.5 \
    --draw-reg               0.5 \
    --soft-drawness-targets \
    --weight-decay           5e-4 \
    --lr                     3e-4 \
    --batch-size             512 \
    --epochs                 30 \
    --geo-patience           5

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
