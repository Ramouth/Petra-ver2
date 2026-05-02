#!/bin/bash
set -euo pipefail
#BSUB -J probe_policy_dr03
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_policy_dr03_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_policy_dr03_%J.err

# Policy-head probe on dr03 (petra-drawness champion).
#
# Diagnoses where the raw 4096-dim logit mass actually lives: on legal
# moves or on illegal ones. Inference is masked but training was not,
# so the head may have learned nothing about legality.
#
# Headline numbers to read out:
#   illegal_mass p50 — fraction of raw softmax mass on illegal moves
#                      >0.95 = DEAD, >0.5 = LEAKY, <0.5 = LEGAL-AWARE
#   masked_entropy / log(n_legal) p50 — normalised entropy of masked policy
#                      >0.9 = UNDIRECTED, <0.2 = COLLAPSED
#
# These two numbers + the per-phase breakdown decide §2 design:
#   DEAD + COLLAPSED       → cold-init new policy head, full retrain
#   DEAD + UNDIRECTED      → cold-init, MCTS-style targets are load-bearing
#   LEAKY + CONTEXTUAL     → fine-tune existing head with masked loss
#   LEGAL-AWARE + anything → keep head, just add masking + MCTS targets
#
# Pair with probe_policy_natural.sh — they're independent, run in parallel:
#   bsub < jobs/probe_policy_dr03.sh
#   bsub < jobs/probe_policy_natural.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODEL="${HOME_DIR}/Petra-ver2/models/drawness_full_natural_init_dr03/best.pt"
DATASET="/dtu/blackhole/0b/206091/dataset_drawness_curriculum.pt"
OUT="${HOME_DIR}/Petra-ver2/results/probe_policy_dr03.json"

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
