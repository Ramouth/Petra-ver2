#!/bin/bash
set -euo pipefail
#BSUB -J probe_policy_natural
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_policy_natural_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_policy_natural_%J.err

# Policy-head probe on natural (petra-elo champion).
#
# Same probe as probe_policy_dr03.sh — gives a comparison point so we
# know whether policy state correlates with the value-geometry split
# between the two forks, or whether both heads are equally dead/alive.
#
# Same dataset (dataset_drawness_curriculum.pt) as the dr03 probe so
# the numbers are directly comparable.
#
# Pair with probe_policy_dr03.sh — they're independent, run in parallel:
#   bsub < jobs/probe_policy_dr03.sh
#   bsub < jobs/probe_policy_natural.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODEL="${HOME_DIR}/Petra-ver2/models/natural/best.pt"
DATASET="/dtu/blackhole/0b/206091/dataset_drawness_curriculum.pt"
OUT="${HOME_DIR}/Petra-ver2/results/probe_policy_natural.json"

if [ ! -f "${MODEL}" ];   then echo "ERROR: not found: ${MODEL}";   exit 1; fi
if [ ! -f "${DATASET}" ]; then echo "ERROR: not found: ${DATASET}"; exit 1; fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

mkdir -p "$(dirname "${OUT}")"

python3 -u "${SRC}/probe_policy.py" \
    --model   "${MODEL}" \
    --dataset "${DATASET}" \
    --n       1000 \
    --out     "${OUT}"

echo "Done. Results: ${OUT}"
