#!/bin/bash
#BSUB -J train_round2
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_round2_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_round2_%J.err

# Train on merged 2025-01 round 2 dataset (chunks 0-7, ~2M positions).
#
# Init from current best geometry checkpoint. Override via:
#   bsub -env "INIT_MODEL=/path/to/better.pt" < jobs/train_round2_gpu.sh
#
# Gates (probe_geometry after training):
#   rank > 21.6   (beat lichess_2023_03_endgame)
#   wdcos < 0.5
#   KR vs KR |value| < 0.35

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${DATASET:-${BLACKHOLE}/dataset_2025_01_sf18_r2.pt}"
INIT_MODEL="${INIT_MODEL:-${HOME_DIR}/Petra-ver2/models/lichess_2023_03_drawness/best.pt}"
ANCHOR_DATASET="${ANCHOR_DATASET:-${HOME_DIR}/Petra-ver2/data/endgame_anchor.pt}"
OUT_DIR="${OUT_DIR:-${HOME_DIR}/Petra-ver2/models/2025_01_r2}"

echo "=== Round 2 training (2025-01, chunks 0-7) ==="
echo "Dataset:    ${DATASET}"
echo "Init model: ${INIT_MODEL}"
echo "Anchor:     ${ANCHOR_DATASET} (frac=0.12)"
echo "Out:        ${OUT_DIR}"
echo

if [ ! -f "${DATASET}" ]; then
    echo "ERROR: dataset not found: ${DATASET}"
    echo "Run: bsub -env \"MONTH=01,YEAR=2025,N_CHUNKS_DONE=8,ROUND=2\" < jobs/reeval_merge.sh"
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
    --init-model     "${INIT_MODEL}" \
    --anchor-dataset "${ANCHOR_DATASET}" \
    --anchor-frac    0.12 \
    --out            "${OUT_DIR}" \
    --rank-reg       0.1 \
    --draw-reg       0.005 \
    --weight-decay   5e-4 \
    --lr             3e-4 \
    --epochs         30 \
    --geo-patience   5 \
    --num-workers    0

echo
echo "Done. Checkpoint: ${OUT_DIR}/best.pt"
