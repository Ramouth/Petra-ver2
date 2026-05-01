#!/bin/bash
#BSUB -J eval_drawness_full_natural_init_vs_natural_dr03
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_drawness_full_natural_init_vs_natural_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_drawness_full_natural_init_vs_natural_%J.err

# Head-to-head: drawness_full_natural_init (EXP 2) vs natural.
#
# This is the honest ELO gate. natural is the current ELO baseline (+143
# vs 2021_06_all). EXP 2 is hypothesised to maintain or improve on that
# while ALSO winning PoC battery (drawness_full's 0.67) and drawness gates
# (drawness_full's gap +0.778).
#
# Pass: wr ≥ 50% — EXP 2 is the new project baseline.
# Fail: wr < 45% — recipe trade-off is real; revisit hypotheses.
# Marginal (45-50%): publishable PoC win, ELO holds — acceptable per
#                    feedback_elo_vs_predictive (predictive ≠ playing).
#
#   bsub < jobs/eval_drawness_full_natural_init_vs_natural.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODELS="${HOME_DIR}/Petra-ver2/models"

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

echo "=== drawness_full_natural_init vs natural (head-to-head, both have policy) ==="
python3 -u "${SRC}/evaluate.py" \
    --model          "${MODELS}/drawness_full_natural_init_dr03/best.pt" \
    --baseline-model "${MODELS}/natural/best.pt" \
    --step 5 \
    --n-sim 100 \
    --games 200 \
    --workers 16
