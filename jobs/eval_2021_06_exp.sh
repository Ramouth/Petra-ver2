#!/bin/bash
#BSUB -J eval_2021_06_${EXP:-mid}
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 6:00
#BSUB -o /zhome/81/b/206091/logs/eval_2021_06_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_2021_06_%J.err

# Evaluate a 2021-06 training experiment: probe geometry + head-to-head ELO.
# EXP controls which model to evaluate: mid | low_mid | all
#
#   bsub -env "EXP=mid"     < jobs/eval_2021_06_exp.sh
#   bsub -env "EXP=low_mid" < jobs/eval_2021_06_exp.sh
#   bsub -env "EXP=all"     < jobs/eval_2021_06_exp.sh

EXP="${EXP:-mid}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

MODEL="${HOME_DIR}/Petra-ver2/models/2021_06_${EXP}/best.pt"
BASELINE="${HOME_DIR}/Petra-ver2/models/lichess_2023_03_endgame/best.pt"
PROBE_DATASET="${BLACKHOLE}/dataset_2021_06_mid_sf18.pt"

echo "=== Eval 2021-06 exp=${EXP} ==="
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
echo "STEP 2 — Head-to-head vs endgame (rank 21.6)"
echo "========================================"
python3 -u "${SRC}/evaluate.py" \
    --model          "${MODEL}" \
    --baseline-model "${BASELINE}" \
    --step           5 \
    --n-sim          100 \
    --games          200 \
    --workers        16

echo
echo "=== Done: eval_2021_06 exp=${EXP} ==="
