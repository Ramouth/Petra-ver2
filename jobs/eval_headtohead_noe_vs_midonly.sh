#!/bin/bash
#BSUB -J eval_hth_noe_vs_midonly
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 6:00
#BSUB -o /zhome/81/b/206091/logs/eval_hth_noe_vs_midonly_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_hth_noe_vs_midonly_%J.err

#   bsub < jobs/eval_headtohead_noe_vs_midonly.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

MODEL="${HOME_DIR}/Petra-ver2/models/phase15_mid_no_endgame/best.pt"
BASELINE="${HOME_DIR}/Petra-ver2/models/phase15_mid_only/best.pt"

echo "=== Head-to-head: mid_no_endgame vs mid_only ==="
echo "Model:    ${MODEL}"
echo "Baseline: ${BASELINE}"
echo "Games:    400  (n_sim=100)"
echo

for F in "${MODEL}" "${BASELINE}"; do
    if [ ! -f "${F}" ]; then echo "ERROR: not found: ${F}"; exit 1; fi
done

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/evaluate.py" \
    --model          "${MODEL}" \
    --baseline-model "${BASELINE}" \
    --step           5 \
    --n-sim          100 \
    --games          400 \
    --workers        16

echo
echo "=== Done: eval_hth_noe_vs_midonly ==="
