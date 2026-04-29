#!/bin/bash
set -euo pipefail
#BSUB -J train_drawness_head_v2
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 8:00
#BSUB -o /zhome/81/b/206091/logs/train_drawness_head_v2_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_drawness_head_v2_%J.err

# Frozen-backbone drawness head training with LR initialisation.
#
# Probe showed the frozen phase15 backbone already separates structural draws
# linearly (Cohen's d=2.33, LR accuracy=0.966) — the head just needs to be
# pointed at the right direction. Random init wasted 16 epochs converging.
#
# Changes vs train_drawness_head.sh:
#   --init-drawness-from-probe  fits LR on training geometry before epoch 1
#   --lr 3e-4                   lower LR — starting near solution, not random
#   --geo-patience 5            fast convergence expected, no need to wait 10
#
#   bsub < jobs/train_drawness_head_v2.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_drawness_curriculum.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/phase15_mid_no_endgame/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/drawness_head_v2"

echo "=== Drawness head v2 (frozen backbone + LR init) ==="
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
    --dataset          "${DATASET}" \
    --init-model       "${INIT_MODEL}" \
    --out              "${OUT_DIR}" \
    --epochs           30 \
    --batch-size       512 \
    --lr               3e-4 \
    --geo-patience     5 \
    --draw-reg         1.0 \
    --freeze-backbone \
    --init-drawness-from-probe

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
echo "Next: bsub -env \"VARIANT=head_v2\" < jobs/probe_drawness_head.sh"
