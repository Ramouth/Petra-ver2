#!/bin/bash
#BSUB -J eval_dec_probe_${DEC_FRAC:-0.50}
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 6:00
#BSUB -o /zhome/81/b/206091/logs/eval_dec_probe_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_dec_probe_%J.err

# Eval dec probe condition: geometry probe + head-to-head vs feb_sf.
# All 5 can run in parallel once their respective train jobs complete.
#
#   bsub -env "DEC_FRAC=0.00" < jobs/eval_dec_probe.sh
#   bsub -env "DEC_FRAC=0.25" < jobs/eval_dec_probe.sh
#   bsub -env "DEC_FRAC=0.50" < jobs/eval_dec_probe.sh
#   bsub -env "DEC_FRAC=0.75" < jobs/eval_dec_probe.sh
#   bsub -env "DEC_FRAC=1.00" < jobs/eval_dec_probe.sh

DEC_FRAC="${DEC_FRAC:-0.50}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

TAG="${DEC_FRAC//./}"
MODEL="${HOME_DIR}/Petra-ver2/models/dec_probe_frac${TAG}/best.pt"
BASELINE="${HOME_DIR}/Petra-ver2/models/feb_sf/best.pt"
PROBE_DATASET="${BLACKHOLE}/dataset_2021_06_mid_sf18.pt"

echo "=== Eval dec probe: DEC_FRAC=${DEC_FRAC} ==="
echo "Model:         ${MODEL}"
echo "Baseline:      ${BASELINE}"
echo "Probe dataset: ${PROBE_DATASET}"
echo

for F in "${MODEL}" "${BASELINE}" "${PROBE_DATASET}"; do
    if [ ! -f "${F}" ]; then echo "ERROR: not found: ${F}"; exit 1; fi
done

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

echo "========================================"
echo "STEP 1 — Geometry probe"
echo "========================================"
python3 -u "${SRC}/probe_geometry.py" \
    --model   "${MODEL}" \
    --dataset "${PROBE_DATASET}" \
    --n       5000

echo
echo "========================================"
echo "STEP 2 — Head-to-head vs feb_sf"
echo "========================================"
python3 -u "${SRC}/evaluate.py" \
    --model          "${MODEL}" \
    --baseline-model "${BASELINE}" \
    --step           5 \
    --n-sim          100 \
    --games          200 \
    --workers        16

echo
echo "=== Done: eval_dec_probe DEC_FRAC=${DEC_FRAC} ==="
