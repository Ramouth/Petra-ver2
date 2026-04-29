#!/bin/bash
#BSUB -J eval_draw_reg_${DRAW_REG:-0.0}
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 6:00
#BSUB -o /zhome/81/b/206091/logs/eval_draw_reg_ablation_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_draw_reg_ablation_%J.err

# Head-to-head eval for one draw-reg ablation condition vs phase15_mid_no_endgame.
#
#   bsub -env "DRAW_REG=0.0" < jobs/eval_draw_reg_ablation.sh
#   bsub -env "DRAW_REG=0.2" < jobs/eval_draw_reg_ablation.sh
#   bsub -env "DRAW_REG=0.5" < jobs/eval_draw_reg_ablation.sh
#   bsub -env "DRAW_REG=1.0" < jobs/eval_draw_reg_ablation.sh
#   bsub -env "DRAW_REG=2.0" < jobs/eval_draw_reg_ablation.sh

DRAW_REG="${DRAW_REG:-0.0}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

TAG="${DRAW_REG//./p}"
MODEL="${HOME_DIR}/Petra-ver2/models/draw_reg_abl/dr${TAG}/best.pt"
BASELINE="${HOME_DIR}/Petra-ver2/models/phase15_mid_no_endgame/best.pt"

echo "=== Draw-reg ablation eval: DRAW_REG=${DRAW_REG} ==="
echo "Model:    ${MODEL}"
echo "Baseline: ${BASELINE}"
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
    --games          200 \
    --workers        16

echo
echo "=== Done: eval_draw_reg_ablation DRAW_REG=${DRAW_REG} ==="
