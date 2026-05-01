#!/bin/bash
set -euo pipefail
#BSUB -J poc_eval_drawness
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/poc_eval_drawness_%J.out
#BSUB -e /zhome/81/b/206091/logs/poc_eval_drawness_%J.err

# PoC battery on the drawness-trained models that were never PoC-evaluated.
# Question: do trained drawness heads (gap +0.7+) translate to geometry-level
# PoC signal? Or are head and geometry decoupled?
#
# Reference baselines (from prior poc_battery_eval_28329572):
#   natural:    centroid AUC=0.589 d=+0.32  logreg AUC=0.607 d=+0.38
#   natural_v2: centroid AUC=0.593 d=+0.34  logreg AUC=0.603 d=+0.40
#
#   bsub < jobs/poc_battery_eval_drawness.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODELS="${HOME_DIR}/Petra-ver2/models"
BATTERY="${HOME_DIR}/Petra-ver2/data/poc_battery_v0.pt"

if [ ! -f "${BATTERY}" ]; then
    echo "ERROR: battery not found: ${BATTERY}"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

for MODEL_NAME in drawness_full drawness_head_v2 drawness_head; do
    MODEL="${MODELS}/${MODEL_NAME}/best.pt"
    if [ ! -f "${MODEL}" ]; then
        echo "WARNING: skipping ${MODEL_NAME} — checkpoint not found at ${MODEL}"
        continue
    fi
    echo
    echo "######################################################################"
    echo "##  ${MODEL_NAME}"
    echo "######################################################################"
    python3 -u "${SRC}/poc_battery_eval.py" \
        --model    "${MODEL}" \
        --battery  "${BATTERY}" \
        --n-folds  5 \
        --seed     42
done

echo
echo "Done."
