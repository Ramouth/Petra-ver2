#!/bin/bash
set -euo pipefail
#BSUB -J poc_eval_big_2025_filtered
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/poc_eval_big_2025_filtered_%J.out
#BSUB -e /zhome/81/b/206091/logs/poc_eval_big_2025_filtered_%J.err

# PoC drawness gate on big_2025_filtered.
#
# Two checkpoints worth comparing:
#   - best.pt    — saved at epoch 1 (rank 25.4, wdcos 0.6651). Save logic
#                  triggers on rank-or-draw improvement; with draw_reg=0
#                  and rank only ever falling after epoch 1, this is what
#                  was kept.
#   - epoch_09.pt — best wdcos of the run (0.2357), rank 19.2. The actual
#                  draw-geometry peak per the per-epoch checkpoint patch.
#
# Compare both against natural's AUC≈0.6 baseline. Honest gate question:
# does the piece-count refinement translate from wdcos to PoC signal?
#
#   bsub < jobs/poc_battery_eval_big_2025_filtered.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODELS="${HOME_DIR}/Petra-ver2/models/big_2025_filtered"
BATTERY="${HOME_DIR}/Petra-ver2/data/poc_battery_v0.pt"

if [ ! -f "${BATTERY}" ]; then
    echo "ERROR: battery not found: ${BATTERY}"
    echo "Run jobs/build_poc_battery.sh first."
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

for CKPT in best.pt epoch_09.pt; do
    MODEL="${MODELS}/${CKPT}"
    if [ ! -f "${MODEL}" ]; then
        echo "WARNING: skipping ${CKPT} — not found at ${MODEL}"
        continue
    fi
    echo
    echo "######################################################################"
    echo "##  big_2025_filtered/${CKPT}"
    echo "######################################################################"
    python3 -u "${SRC}/poc_battery_eval.py" \
        --model    "${MODEL}" \
        --battery  "${BATTERY}" \
        --n-folds  5 \
        --seed     42
done

echo
echo "Done."
