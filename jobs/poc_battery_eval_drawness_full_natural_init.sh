#!/bin/bash
set -euo pipefail
#BSUB -J poc_eval_drawness_full_natural_init
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/poc_eval_drawness_full_natural_init_%J.out
#BSUB -e /zhome/81/b/206091/logs/poc_eval_drawness_full_natural_init_%J.err

# PoC battery on drawness_full_natural_init (EXP 2 result).
#
# Reference baselines:
#   natural:        centroid AUC=0.589  pc1=FAIL  logreg=0.607  d=+0.32
#   drawness_full:  centroid AUC=0.670  pc1=PASS  logreg=0.666  d=+0.65  (project best)
#
# EXP 2 prediction (PoC):
#   - centroid AUC ≥ 0.65 (BCE+draw_reg rotates PC1)
#   - PC1 PASSES (the new bar — natural's PC1 fails)
#   - d ≥ 0.55
# A pass here means EXP 2 cracked the project on the geometry side.
#
#   bsub < jobs/poc_battery_eval_drawness_full_natural_init.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODEL="${HOME_DIR}/Petra-ver2/models/drawness_full_natural_init/best.pt"
BATTERY="${HOME_DIR}/Petra-ver2/data/poc_battery_v0.pt"

if [ ! -f "${MODEL}" ];   then echo "ERROR: not found: ${MODEL}";   exit 1; fi
if [ ! -f "${BATTERY}" ]; then echo "ERROR: not found: ${BATTERY}"; exit 1; fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

echo "######################################################################"
echo "##  drawness_full_natural_init"
echo "######################################################################"
python3 -u "${SRC}/poc_battery_eval.py" \
    --model    "${MODEL}" \
    --battery  "${BATTERY}" \
    --n-folds  5 \
    --seed     42

echo "Done."
