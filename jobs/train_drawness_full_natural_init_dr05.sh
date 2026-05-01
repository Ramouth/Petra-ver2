#!/bin/bash
set -euo pipefail
#BSUB -J drawness_full_natural_init_dr05
#BSUB -q gpul40s
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/drawness_full_natural_init_dr05_%J.out
#BSUB -e /zhome/81/b/206091/logs/drawness_full_natural_init_dr05_%J.err

# Path B — squaring the circle by tuning draw_reg strength.
#
# drawness_full_natural_init at draw_reg=1.0 rotated PC1 to drawness but
# at the cost of win/loss separation (cos -0.65 -> +0.10) and rank
# (38.7 -> 23.1). The rotation was too strong.
#
# Hypothesis: the drawness/win-loss tension is continuous. At lower
# draw_reg, PC1 rotates partially — drawness signal still strong enough
# for PoC (≥0.62) but win/loss axis preserved enough for ELO.
#
# Reference recipe: train_drawness_full_natural_init.sh — only --draw-reg
# changed (1.0 -> 0.5, half strength).
#
# Predicted outcome (with H1+H2 from the loop):
#   - rank 28-32 (between drawness_full_natural_init's 23 and natural's 38)
#   - PoC AUC 0.63-0.66 (probably 3/3 still pass)
#   - win·loss strict cosine -0.30 to -0.50 (partial preservation)
#   - Top1 0.20-0.22 (within range)
#   - ELO H2H vs natural: closer to 50% than EXP 2's 39.8%
#
#   bsub < jobs/train_drawness_full_natural_init_dr05.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_drawness_curriculum.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/2021_06_all/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/drawness_full_natural_init_dr05"

echo "=== Path B (1/2): draw_reg=0.5 ==="
echo "Dataset:    ${DATASET}"
echo "Init model: ${INIT_MODEL}"
echo "Out:        ${OUT_DIR}"
echo "Recipe:     BCE+draw_reg+curriculum, lr 3e-4, draw-reg 0.5 (was 1.0)"
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
    --policy-weight     1.0 \
    --rank-reg          0.5 \
    --draw-reg          0.5 \
    --weight-decay      5e-4 \
    --lr                3e-4 \
    --epochs            20 \
    --geo-patience      5 \
    --num-workers       0

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
