#!/bin/bash
set -euo pipefail
#BSUB -J drawness_full_natural_init_dr03
#BSUB -q gpul40s
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/drawness_full_natural_init_dr03_%J.out
#BSUB -e /zhome/81/b/206091/logs/drawness_full_natural_init_dr03_%J.err

# Path B — squaring the circle by tuning draw_reg strength.
#
# Lower-bound variant: draw_reg=0.3 (was 1.0 in EXP 2).
#
# Tests the lower end of the Pareto sweep. If draw_reg=0.5 still rotates
# PC1 too much, 0.3 might preserve more win/loss separation while still
# putting enough drawness signal in geometry for PoC battery to pass.
#
# Pair with dr05 to map the draw_reg → {rank, PoC, ELO} trade curve.
#
#   bsub < jobs/train_drawness_full_natural_init_dr03.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_drawness_curriculum.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/2021_06_all/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/drawness_full_natural_init_dr03"

echo "=== Path B (2/2): draw_reg=0.3 ==="
echo "Dataset:    ${DATASET}"
echo "Init model: ${INIT_MODEL}"
echo "Out:        ${OUT_DIR}"
echo "Recipe:     BCE+draw_reg+curriculum, lr 3e-4, draw-reg 0.3 (was 1.0)"
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
    --draw-reg          0.3 \
    --weight-decay      5e-4 \
    --lr                3e-4 \
    --epochs            20 \
    --geo-patience      5 \
    --num-workers       0

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
