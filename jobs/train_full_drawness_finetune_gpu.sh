#!/bin/bash
#BSUB -J train_full_drawness_ft
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 12:00
#BSUB -o /zhome/81/b/206091/logs/train_full_drawness_ft_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_full_drawness_ft_%J.err

# Fine-tune from the regular full-dataset checkpoint on an SF-labelled dataset
# that also contains soft drawn-game drawness targets.
#
# Keep draw_reg small: this is a distribution-broadening draw prior, not a hard
# theoretical-draw bootstrap.

BLACKHOLE="${BLACKHOLE:-/dtu/blackhole/0b/206091}"
HOME_DIR="${HOME_DIR:-/zhome/81/b/206091}"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${DATASET:-${BLACKHOLE}/dataset_2025_01_sf18_drawness.pt}"
INIT_MODEL="${INIT_MODEL:-${HOME_DIR}/Petra-ver2/models/full_regular/best.pt}"
OUT_DIR="${OUT_DIR:-${HOME_DIR}/Petra-ver2/models/full_drawness_finetune}"

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
RANK_REG="${RANK_REG:-0.05}"
DRAW_REG="${DRAW_REG:-0.005}"
EPOCHS="${EPOCHS:-10}"
GEO_PATIENCE="${GEO_PATIENCE:-4}"

echo "=== Full dataset drawness fine-tune ==="
echo "Dataset:    ${DATASET}"
echo "Init model: ${INIT_MODEL}"
echo "Out:        ${OUT_DIR}"
echo

if [ ! -f "${DATASET}" ]; then
    echo "ERROR: dataset not found: ${DATASET}"
    exit 1
fi

if [ ! -f "${INIT_MODEL}" ]; then
    echo "ERROR: init model not found: ${INIT_MODEL}"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

python3 -u "${SRC}/train.py" \
    --dataset      "${DATASET}" \
    --init-model   "${INIT_MODEL}" \
    --out          "${OUT_DIR}" \
    --rank-reg     "${RANK_REG}" \
    --draw-reg     "${DRAW_REG}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --lr           "${LR}" \
    --epochs       "${EPOCHS}" \
    --geo-patience "${GEO_PATIENCE}" \
    --num-workers  0

echo
echo "Done. Fine-tuned checkpoint: ${OUT_DIR}/best.pt"
