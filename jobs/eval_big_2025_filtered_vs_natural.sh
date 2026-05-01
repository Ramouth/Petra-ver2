#!/bin/bash
set -euo pipefail
#BSUB -J eval_big_2025_filtered_vs_natural
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_big_2025_filtered_vs_natural_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_big_2025_filtered_vs_natural_%J.err

# Head-to-head: big_2025_filtered (best.pt) vs natural.
#
# Expected result: big_2025_filtered loses heavily. Top1 collapsed from
# 0.31 (natural) to 0.02 over training — policy head is effectively
# random. MCTS without a usable policy prior will lose to MCTS with one,
# regardless of the value head's calibration.
#
# We run it anyway as the honest ELO gate: ELO is the project's secondary
# but live goal. If it's a complete blowout (>80%+ for natural) it
# confirms the policy collapse is the bottleneck, not the value head.
#
#   bsub < jobs/eval_big_2025_filtered_vs_natural.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODELS="${HOME_DIR}/Petra-ver2/models"

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

echo "=== big_2025_filtered (best.pt) vs natural (head-to-head) ==="
python3 -u "${SRC}/evaluate.py" \
    --model          "${MODELS}/big_2025_filtered/best.pt" \
    --baseline-model "${MODELS}/natural/best.pt" \
    --step 5 \
    --n-sim 100 \
    --games 200 \
    --workers 16
