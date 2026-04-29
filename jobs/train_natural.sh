#!/bin/bash
set -euo pipefail
#BSUB -J train_natural
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_natural_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_natural_%J.err

# "Natural" drawness experiment — no drawness scaffolding.
#
# Hypothesis: the curriculum's value distribution (76k drawn + 400k decisive)
# alone is enough to shape geometry into draw-aware regions. Explicit
# drawness BCE was scaffolding; if the geometry hypothesis is right,
# value MSE + rank-reg + policy will self-organise the same structure
# without surface-feature overfitting risk.
#
# Init from 2021_06_all (rank 87, has trained policy from Phase 1) — best
# starting point we have. Drops --policy-weight 0 (which broke ELO eval)
# and --draw-reg (which we now suspect is unnecessary).
#
#   bsub < jobs/train_natural.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_drawness_curriculum.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/2021_06_all/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/natural"

echo "=== Natural drawness experiment (no drawness scaffolding) ==="
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
    --dataset       "${DATASET}" \
    --init-model    "${INIT_MODEL}" \
    --out           "${OUT_DIR}" \
    --policy-weight 1.0 \
    --rank-reg      0.5 \
    --draw-reg      0.0 \
    --weight-decay  5e-4 \
    --lr            3e-4 \
    --batch-size    512 \
    --epochs        30 \
    --geo-patience  5

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
echo "Next: bsub -env \"VARIANT=natural\" < jobs/probe_drawness_head.sh"
echo "      bsub < jobs/eval_natural_vs_feb_sf.sh   # fair fight, both have policy"
