#!/bin/bash
set -euo pipefail
#BSUB -J drawness_full_natural_init
#BSUB -q gpul40s
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/drawness_full_natural_init_%J.out
#BSUB -e /zhome/81/b/206091/logs/drawness_full_natural_init_%J.err

# EXP 2 from the loop's Phase 3 hypothesis:
# drawness_full's recipe (BCE λ=1 + draw_reg + curriculum) but with two
# variables changed from the original train_drawness_full.sh:
#
#   --init-model:     phase15_mid_no_endgame -> 2021_06_all
#   --policy-weight:  0.0                    -> 1.0
#
# Why these two: drawness_full passed PoC battery decisively (centroid AUC
# 0.670 vs natural's 0.589, PC1 also passed) — the BCE+draw_reg recipe DID
# shape geometry. But it was paired with a policy-disabling init AND
# explicit policy-weight=0, leaving the policy dead (Top1 0.085 throughout)
# and ELO at -100 vs feb_sf.
#
# Predicted under loop H1+H2:
#   - PoC AUC ≥ 0.65 (BCE+draw_reg rotated PC1 toward drawness)
#   - Top1 ≥ 0.22 (healthy 2021_06_all policy + policy-weight 1.0 keeps it)
#   - rank ≥ 30 (natural was 38.7, drawness_full was 34.2)
#   - Drawness head gap ≥ 0.6 (drawness_full reached 0.778)
#   - ELO H2H natural: ≥ 50% wr → strict improvement
#
# All other recipe parameters identical to drawness_full.
#
#   bsub < jobs/train_drawness_full_natural_init.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_drawness_curriculum.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/2021_06_all/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/drawness_full_natural_init"

echo "=== EXP 2: drawness_full recipe + healthy policy init ==="
echo "Dataset:    ${DATASET}"
echo "Init model: ${INIT_MODEL}  (was phase15_mid_no_endgame)"
echo "Out:        ${OUT_DIR}"
echo "Recipe:     BCE+draw_reg+curriculum, lr 3e-4, policy-weight 1.0 (was 0.0)"
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
    --draw-reg          1.0 \
    --weight-decay      5e-4 \
    --lr                3e-4 \
    --epochs            20 \
    --geo-patience      5 \
    --num-workers       0

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
echo
echo "Next steps (in parallel):"
echo "  bsub < jobs/poc_battery_eval_drawness.sh   # add this model to the loop and re-run"
echo "  bsub -env \"VARIANT=full_natural_init\" -env \"MODEL_DIR=drawness_full_natural_init\" \\"
echo "       < jobs/probe_drawness_head.sh"
echo "  bsub < jobs/probe_geometry_natural.sh      # adapt for this model"
echo "  bsub < jobs/eval_natural_vs_feb_sf.sh      # ELO H2H, edit MODEL"
