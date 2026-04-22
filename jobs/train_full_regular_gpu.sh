#!/bin/bash
#BSUB -J train_full_regular
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_full_regular_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_full_regular_%J.err

# Regular broad training on an already SF-labelled full dataset.
#
# Use this now while the drawn-game drawness dataset is being re-parsed and
# re-evaluated in parallel. This phase trains value/policy/geometry only; it
# does not need outcome_values/plys metadata.
#
# Override example:
#   bsub -env "DATASET=/dtu/blackhole/0b/206091/dataset_2025_01_sf18_r1.pt,INIT_MODEL=/zhome/81/b/206091/Petra-ver2/models/drawness_bootstrap/best.pt,OUT_DIR=/zhome/81/b/206091/Petra-ver2/models/full_regular" < jobs/train_full_regular_gpu.sh

BLACKHOLE="${BLACKHOLE:-/dtu/blackhole/0b/206091}"
HOME_DIR="${HOME_DIR:-/zhome/81/b/206091}"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${DATASET:-${BLACKHOLE}/dataset_2025_01_sf18_r1.pt}"
INIT_MODEL="${INIT_MODEL:-${HOME_DIR}/Petra-ver2/models/lichess_2023_03_endgame/best.pt}"
OUT_DIR="${OUT_DIR:-${HOME_DIR}/Petra-ver2/models/full_regular}"

LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
RANK_REG="${RANK_REG:-0.1}"
EPOCHS="${EPOCHS:-50}"
GEO_PATIENCE="${GEO_PATIENCE:-5}"

echo "=== Regular full-dataset training ==="
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
    --weight-decay "${WEIGHT_DECAY}" \
    --lr           "${LR}" \
    --epochs       "${EPOCHS}" \
    --geo-patience "${GEO_PATIENCE}" \
    --num-workers  0

echo
echo "Done. Regular checkpoint: ${OUT_DIR}/best.pt"
