#!/bin/bash
set -euo pipefail
#BSUB -J train_big
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=64GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_big_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_big_%J.err

# Train `big`: outcome-supervised model on 5M positions from a fresh
# 2025-04 parse at min_elo=2400. Init from natural — the strongest
# baseline by ELO + the chosen lineage.
#
# Hypothesis: outcome supervision (game result, not SF eval) on a much
# larger corpus pushes the drawness probe (poc_battery_eval) above
# natural's 58%. Because the loss directly targets game outcome, the
# representation must encode features that distinguish |SF|≈0 drawn
# games from |SF|≈0 decisive games — exactly the PoC capability.
#
# No drawness scaffolding (no draw_reg, no BCE) — same minimal recipe
# as natural. Patience 10 because larger data needs longer reshape.
#
#   bsub < jobs/train_big.sh   # after build_big.sh completes

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_big.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/natural/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/big"

echo "=== Train big (outcome supervision, 5M @ 7.5% drawn, init from natural) ==="
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
    --epochs        25 \
    --geo-patience  10

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
echo "Next: bsub < jobs/poc_battery_eval.sh   (after editing to include big)"
