#!/bin/bash
#BSUB -J fit_drawness_head_natural
#BSUB -q gpul40s
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/fit_drawness_head_natural_%J.out
#BSUB -e /zhome/81/b/206091/logs/fit_drawness_head_natural_%J.err

# Post-hoc fit drawness head on natural's frozen backbone.
#
# Question: is the drawness signal already in natural's geometry, just
# unread? natural was trained on drawness_curriculum without scaffolding
# (no BCE), so its drawness head returns constant ~0.5 (probe 28327737).
# fit_drawness_head.py runs sklearn L-BFGS to find the global BCE optimum
# from synthetic structural-draw vs decisive-endgame embeddings.
#
# Reference points:
#   - drawness_head     (post-hoc on phase15_mid_no_endgame): 2/4 gates
#   - drawness_head_v2  (post-hoc on phase15_mid_no_endgame v2): 3/4 gates
#   - drawness_full     (full-train + BCE): 3/4 gates, KR=0.808
#
# If natural_with_drawness gets ≥3/4 gates → drawness signal IS in natural's
# geometry, just unread. That changes the project plan: the lever is the
# readout, not new training.
# If natural_with_drawness gets ≤2/4 → geometry-level signal is missing,
# need a different training intervention.
#
#   bsub < jobs/fit_drawness_head_natural.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODELS="${HOME_DIR}/Petra-ver2/models"

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

python3 -u "${SRC}/fit_drawness_head.py" \
    --model "${MODELS}/natural/best.pt" \
    --out   "${MODELS}/natural_with_drawness/best.pt" \
    --n     50000 \
    --C     1.0

echo
echo "Now probe the result:"
echo "  bsub -env \"VARIANT=with_drawness\" -env \"MODEL_DIR=natural_with_drawness\" \\"
echo "       < jobs/probe_drawness_head.sh"
