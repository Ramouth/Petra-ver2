#!/bin/bash
set -euo pipefail
#BSUB -J train_drawness_head
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 8:00
#BSUB -o /zhome/81/b/206091/logs/train_drawness_head_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_drawness_head_%J.err

# Frozen-backbone drawness head training on curriculum dataset.
# Backbone is fixed (rank preserved), only the 129-parameter drawness
# head receives gradient via the BCE draw-reg loss.
#
# Dataset: 76,851 drawness positives + 400k decisive negatives (16%/84%)
# Decisive positions contribute via draw_neg mask (|v|>0.5 → target=0)
# without needing explicit drawness_mask labels.
#
# Gates (probe after training):
#   KR vs KR drawness > 0.7
#   Sharp balanced drawness < 0.3
#   KNN vs K drawness > 0.7
#   KQ vs K drawness < 0.3
#   Rank unchanged (backbone frozen)
#
#   bsub < jobs/train_drawness_head.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_drawness_curriculum.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/phase15_mid_no_endgame/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/drawness_head"

echo "=== Drawness head training (frozen backbone) ==="
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
    --dataset           "${DATASET}" \
    --init-model        "${INIT_MODEL}" \
    --out               "${OUT_DIR}" \
    --freeze-backbone \
    --policy-weight     0.0 \
    --rank-reg          0.0 \
    --draw-reg          1.0 \
    --weight-decay      0.0 \
    --lr                1e-3 \
    --epochs            30 \
    --geo-patience      10 \
    --num-workers       0

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
echo "Next: bsub < jobs/probe_drawness_head.sh"
