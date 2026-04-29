#!/bin/bash
set -euo pipefail
#BSUB -J train_drawness_full
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 8:00
#BSUB -o /zhome/81/b/206091/logs/train_drawness_full_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_drawness_full_%J.err

# Unfrozen backbone drawness training on curriculum dataset.
# Both backbone and drawness head receive gradient.
# Rank-reg preserves geometry; draw-reg trains the head and reshapes
# the backbone toward a distinct draw cluster.
#
# Pair with train_drawness_head.sh (frozen) to isolate:
#   - Frozen: can the existing geometry support a linear drawness head?
#   - Unfrozen: does joint training open the draw dimension further?
#
# Gates (probe after training):
#   KR vs KR drawness > 0.7
#   Sharp balanced drawness < 0.3
#   KNN vs K drawness > 0.7
#   KQ vs K drawness < 0.3
#   Rank >= 70 (backbone allowed to change but must not collapse)
#
#   bsub < jobs/train_drawness_full.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_drawness_curriculum.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/phase15_mid_no_endgame/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/drawness_full"

echo "=== Drawness full training (unfrozen backbone) ==="
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
    --policy-weight     0.0 \
    --rank-reg          0.5 \
    --draw-reg          1.0 \
    --weight-decay      5e-4 \
    --lr                3e-4 \
    --epochs            20 \
    --geo-patience      5 \
    --num-workers       0

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
echo "Next: bsub < jobs/probe_drawness_head.sh  (update MODEL path first)"
