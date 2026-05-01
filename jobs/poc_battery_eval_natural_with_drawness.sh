#!/bin/bash
set -euo pipefail
#BSUB -J poc_eval_natural_with_drawness
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/poc_eval_natural_with_drawness_%J.out
#BSUB -e /zhome/81/b/206091/logs/poc_eval_natural_with_drawness_%J.err

# PoC battery on natural + post-hoc fit drawness head.
#
# Squaring the circle: drawness_full_natural_init proved the drawness signal
# is achievable, but at the cost of rank (38.7 -> 23.1) and ELO (-72). The
# model rotated its geometry to put drawness on PC1.
#
# Hypothesis: natural's geometry already has the drawness signal, just NOT
# on PC1. A post-hoc fit drawness head can read it without rotating the
# backbone — keeping rank, win/loss separation, and ELO intact.
#
# Reference points:
#   natural alone:                centroid 0.589  PC1 FAIL 0.493  logreg 0.607  d=+0.32
#   drawness_full_natural_init:   centroid 0.664  PC1 PASS 0.666  logreg 0.665  d=+0.62
#   drawness_head (post-hoc on phase15): centroid 0.648  d=+0.54
#
# Pass criteria for "circle squared":
#   PoC AUC >= 0.62 (matches drawness_full_natural_init within CI)
#   Drawness head gates >= 3/4 (post-hoc fit precedent)
#   Geometry IS natural's (rank 38.7, ELO baseline preserved by construction)
#
#   bsub < jobs/poc_battery_eval_natural_with_drawness.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODEL="${HOME_DIR}/Petra-ver2/models/natural_with_drawness/best.pt"
BATTERY="${HOME_DIR}/Petra-ver2/data/poc_battery_v0.pt"

if [ ! -f "${MODEL}" ];   then echo "ERROR: not found: ${MODEL}";   exit 1; fi
if [ ! -f "${BATTERY}" ]; then echo "ERROR: not found: ${BATTERY}"; exit 1; fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

echo "######################################################################"
echo "##  natural_with_drawness (post-hoc fit head on natural's frozen backbone)"
echo "######################################################################"
python3 -u "${SRC}/poc_battery_eval.py" \
    --model    "${MODEL}" \
    --battery  "${BATTERY}" \
    --n-folds  5 \
    --seed     42

echo "Done."
