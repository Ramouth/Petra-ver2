#!/bin/bash
set -euo pipefail
#BSUB -J merge_2025_chunks
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/merge_2025_chunks_%J.out
#BSUB -e /zhome/81/b/206091/logs/merge_2025_chunks_%J.err

# Merge 3 raw chunks of a single month into one split dataset.
# data.py --merge-raw rebases game_ids across chunks and produces a
# train/val split.
#
#   bsub -env "MONTH=02" < jobs/merge_2025_chunks.sh
#   bsub -env "MONTH=04" < jobs/merge_2025_chunks.sh

MONTH="${MONTH:?MONTH must be set (02 or 04)}"
YEAR=2025

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

CHUNK0="${BLACKHOLE}/dataset_${YEAR}_${MONTH}_chunk0.pt"
CHUNK1="${BLACKHOLE}/dataset_${YEAR}_${MONTH}_chunk1.pt"
CHUNK2="${BLACKHOLE}/dataset_${YEAR}_${MONTH}_chunk2.pt"
OUT="${BLACKHOLE}/dataset_${YEAR}_${MONTH}.pt"

echo "=== Merging ${YEAR}-${MONTH} chunks ==="
for F in "${CHUNK0}" "${CHUNK1}" "${CHUNK2}"; do
    if [ ! -f "${F}" ]; then echo "ERROR: missing chunk: ${F}"; exit 1; fi
    echo "  ${F}  ($(du -h "${F}" | cut -f1))"
done
echo "Out: ${OUT}"
echo

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/data.py" \
    --merge-raw "${CHUNK0}" "${CHUNK1}" "${CHUNK2}" \
    --out       "${OUT}"

echo
echo "Done. Merged dataset: ${OUT}"
echo "After both months merged, run: bsub < jobs/build_big.sh"
