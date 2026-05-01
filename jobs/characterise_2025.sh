#!/bin/bash
#BSUB -J characterise_2025
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=24GB]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/characterise_2025_%J.out
#BSUB -e /zhome/81/b/206091/logs/characterise_2025_%J.err

# Survey the 2025 outcome-supervised datasets used by train_big_2025.sh.
# Reports outcome-class counts and phase distribution within each class —
# answers whether draw labels concentrate in endgame (good for drawness
# geometry) or scatter across all phases (bad).
#
# Submit:
#   bsub < jobs/characterise_2025.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/characterise_2025.py"
