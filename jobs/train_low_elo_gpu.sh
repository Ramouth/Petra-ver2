#!/bin/bash
#BSUB -J train_low_elo
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_low_elo_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_low_elo_%J.err

# Train on Lichess 2025-01 low-ELO (1600-1850) dataset.
#
# Hypothesis: lower-rated games expose the geometry to a wider variety of
# positions (unusual pawn structures, non-theory openings, tactical chaos)
# which forces the bottleneck to span more dimensions → higher effective rank.
#
# Init: lichess_2023_03_drawness/best.pt (rank≈19.5, drawness bootstrapped).
# No anchor — rank recovery is the goal, drawness head is already in place.
#
# Gates (probe_geometry after training):
#   rank > 19.5   (improvement over current best)
#   wdcos < 0.5   (no collapse)
#   KR vs KR |value| < 0.35

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${DATASET:-${BLACKHOLE}/dataset_2025_01_low_elo_sf18.pt}"
INIT_MODEL="${INIT_MODEL:-${HOME_DIR}/Petra-ver2/models/lichess_2023_03_drawness/best.pt}"
OUT_DIR="${OUT_DIR:-${HOME_DIR}/Petra-ver2/models/low_elo_2025_01}"

echo "=== Low-ELO training (1600-1850 band) ==="
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
    --rank-reg     0.1 \
    --weight-decay 5e-4 \
    --lr           3e-4 \
    --epochs       30 \
    --geo-patience 5 \
    --num-workers  0

echo
echo "Done. Checkpoint: ${OUT_DIR}/best.pt"
