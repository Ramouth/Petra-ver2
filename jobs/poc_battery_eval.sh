#!/bin/bash
set -euo pipefail
#BSUB -J poc_battery_eval
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/poc_battery_eval_%J.out
#BSUB -e /zhome/81/b/206091/logs/poc_battery_eval_%J.err

# Run the PoC drawness gate on natural and natural_v2.
# The gate asks: from a dead-even middlegame position alone, can the
# learned geometry distinguish drawn-game from decisive-game outcomes?
#
# Three signals, 5-fold CV, report balanced accuracy + 95% CI + AUC + d.
# Random baseline = 50%. Gate passes if 95% CI lower-bound > 0.5.
#
#   bsub < jobs/poc_battery_eval.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODELS="${HOME_DIR}/Petra-ver2/models"
BATTERY="${HOME_DIR}/Petra-ver2/data/poc_battery_v0.pt"

if [ ! -f "${BATTERY}" ]; then
    echo "ERROR: battery not found: ${BATTERY}"
    echo "Run jobs/build_poc_battery.sh first."
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

for MODEL_NAME in natural natural_v2; do
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
