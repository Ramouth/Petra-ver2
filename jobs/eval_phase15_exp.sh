#!/bin/bash
#BSUB -J eval_phase15_${EXP:-mid_only}
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 6:00
#BSUB -o /zhome/81/b/206091/logs/eval_phase15_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_phase15_%J.err

# Evaluate a Phase 1.5 ablation experiment: geometry probe + head-to-head ELO.
# EXP controls which model to evaluate:
#
#   mid_only        — mid-band data only (baseline condition)
#   special_only    — endgame + near_mate + near_mate_losing + material (no mid)
#   mid_full        — mid + all special datasets
#   mid_no_endgame  — mid + near_mate + near_mate_losing + material
#   mid_no_nearmate — mid + endgame + material
#   mid_no_material — mid + endgame + near_mate + near_mate_losing
#
# Dataset position counts:
#   endgame_phase15_50k:       191,366  (172,230 train / 19,136 val)
#   dataset_near_mate:          63,953  ( 60,756 train /  3,197 val)
#   dataset_near_mate_losing:   32,510  ( 30,885 train /  1,625 val)
#   dataset_material_imbalance: 200,000 (190,000 train / 10,000 val)
#   mid reeval (partial):      ~982,000  (parts 0+1+2, part1 complete)
#
#   bsub -env "EXP=mid_only"        < jobs/eval_phase15_exp.sh
#   bsub -env "EXP=special_only"    < jobs/eval_phase15_exp.sh
#   bsub -env "EXP=mid_full"        < jobs/eval_phase15_exp.sh
#   bsub -env "EXP=mid_no_endgame"  < jobs/eval_phase15_exp.sh
#   bsub -env "EXP=mid_no_nearmate" < jobs/eval_phase15_exp.sh
#   bsub -env "EXP=mid_no_material" < jobs/eval_phase15_exp.sh

EXP="${EXP:-mid_only}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

MODEL="${HOME_DIR}/Petra-ver2/models/phase15_${EXP}/best.pt"
BASELINE="${HOME_DIR}/Petra-ver2/models/feb_sf/best.pt"
PROBE_DATASET="${BLACKHOLE}/dataset_2021_06_mid_sf18.pt"

echo "=== Eval Phase 1.5 exp=${EXP} ==="
echo "Model:         ${MODEL}"
echo "Baseline:      ${BASELINE}  (feb_sf, rank ~18.9, +58 ELO vs sf_balanced)"
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
echo "=== Done: eval_phase15 exp=${EXP} ==="
