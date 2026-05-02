#!/bin/bash
set -euo pipefail
#BSUB -J probe_policy_after_training
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_policy_after_training_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_policy_after_training_%J.err

# Phase 2 §2 sanity gate — re-probe the policy head AFTER frozen-trunk training.
#
# Pre-training baseline (probe_policy_dr03.sh, 2026-05-02):
#   illegal_mass p50         = 0.993   (DEAD — raw mass dominated by illegals)
#   masked norm-entropy p50  = 0.983   (UNDIRECTED — masked policy near uniform)
#   Verdict: DEAD + UNDIRECTED
#
# Expected post-training (gate for §2 done):
#   illegal_mass p50         < 0.05    (masking did its job)
#   masked norm-entropy p50  < 0.5     (head is contextual, not flat)
#   per-phase consistent
#
# Same dataset (dataset_drawness_curriculum.pt) and same n=1000 as the
# pre-training probe so the numbers are directly comparable.
#
#   bsub < jobs/probe_policy_after_training.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODEL="${HOME_DIR}/Petra-ver2/models/policy_dr03_init/best.pt"
DATASET="/dtu/blackhole/0b/206091/dataset_drawness_curriculum.pt"
OUT="${HOME_DIR}/Petra-ver2/results/probe_policy_after_training.json"

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
