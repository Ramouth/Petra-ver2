#!/bin/bash
#BSUB -J train_exp_b
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/train_exp_b_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_exp_b_%J.err

# Experiment B: no policy, rank λ=0.5
#
# Same as Exp A but with 5× stronger rank pressure.
# Tests whether λ=0.1 was simply too weak to compete with the loss landscape
# even after removing policy, or if a stronger signal opens new dimensions.
#
# Gates:
#   rank > 21.6   (any improvement over current best)
#   wdcos < 0.5
#   KR vs KR |value| < 0.35

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${DATASET:-${BLACKHOLE}/dataset_2023_03_sf18.pt}"
EXTRA_DATASET="${EXTRA_DATASET:-${BLACKHOLE}/dataset_2025_01_low_elo_sf18.pt}"
INIT_MODEL="${INIT_MODEL:-${HOME_DIR}/Petra-ver2/models/lichess_2023_03_endgame/best.pt}"
OUT_DIR="${OUT_DIR:-${HOME_DIR}/Petra-ver2/models/exp_b_no_policy_strong}"

echo "=== Exp B: no policy, rank λ=0.5 ==="
echo "Dataset:    ${DATASET} + ${EXTRA_DATASET}"
echo "Init model: ${INIT_MODEL}"
echo "Out:        ${OUT_DIR}"
echo

if [ ! -f "${DATASET}" ]; then
    echo "ERROR: dataset not found: ${DATASET}"
    exit 1
fi

if [ ! -f "${EXTRA_DATASET}" ]; then
    echo "ERROR: extra dataset not found: ${EXTRA_DATASET}"
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
    --dataset        "${DATASET}" \
    --extra-dataset  "${EXTRA_DATASET}" \
    --init-model     "${INIT_MODEL}" \
    --out            "${OUT_DIR}" \
    --policy-weight  0.0 \
    --rank-reg       0.5 \
    --draw-reg       0.0 \
    --weight-decay   5e-4 \
    --lr             3e-4 \
    --epochs         30 \
    --geo-patience   5 \
    --num-workers    0

echo
echo "Done. Checkpoint: ${OUT_DIR}/best.pt"
