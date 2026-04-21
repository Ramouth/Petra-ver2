#!/bin/bash
#BSUB -J split_dataset
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/split_dataset_%J.out
#BSUB -e /zhome/81/b/206091/logs/split_dataset_%J.err

# ── Configuration ──────────────────────────────────────────────────────────────
# Override via -env:
#   bsub -env "MONTH=03,YEAR=2023,DEPTH=18" < jobs/split_dataset.sh

MONTH="${MONTH:-03}"
YEAR="${YEAR:-2023}"
DEPTH="${DEPTH:-18}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
DATA="${HOME_DIR}/Petra-ver2/data"

IN_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}_sf${DEPTH}.pt"
DECISIVE_OUT="${DATA}/decisive_${YEAR}_${MONTH}.pt"
BALANCED_OUT="${DATA}/balanced_${YEAR}_${MONTH}.pt"

echo "=== split_dataset: ${YEAR}-${MONTH} depth=${DEPTH} ==="
echo "In:       ${IN_FILE}"
echo "Decisive: ${DECISIVE_OUT}"
echo "Balanced: ${BALANCED_OUT}"
echo

if [ ! -f "${IN_FILE}" ]; then
    echo "ERROR: input not found: ${IN_FILE}"
    exit 1
fi

mkdir -p "${DATA}"

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/split_dataset.py" \
    --dataset      "${IN_FILE}" \
    --decisive-out "${DECISIVE_OUT}" \
    --balanced-out "${BALANCED_OUT}"

echo
echo "Done."
