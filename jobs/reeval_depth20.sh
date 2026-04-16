#!/bin/bash
#BSUB -J reeval_depth20
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 20:00
#BSUB -o /zhome/81/b/206091/logs/reeval_depth20_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_depth20_%J.err

# ── Configuration ──────────────────────────────────────────────────────────────
# Set MONTH, YEAR, and CHUNK_IDX before submitting.
# Submit one job per chunk (0, 1, 2), all with same MONTH/YEAR:
#
#   MONTH=03 YEAR=2020 CHUNK_IDX=0 bsub < jobs/reeval_depth20.sh
#   MONTH=03 YEAR=2020 CHUNK_IDX=1 bsub < jobs/reeval_depth20.sh
#   MONTH=03 YEAR=2020 CHUNK_IDX=2 bsub < jobs/reeval_depth20.sh
#
# After all 3 finish, run the merge job:
#   MONTH=03 YEAR=2020 bsub < jobs/reeval_merge.sh

MONTH="${MONTH:-03}"
YEAR="${YEAR:-2020}"
CHUNK_IDX="${CHUNK_IDX:-0}"
N_CHUNKS=3          # 400k positions / 3 ≈ 133k per chunk → ~8-10h at depth 20
N=400000
SEED=42

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}.pt"
PARTIAL_OUT="${BLACKHOLE}/reeval_${YEAR}_${MONTH}_d20_part${CHUNK_IDX}.pt"

echo "=== SF depth-20 re-evaluation: ${YEAR}-${MONTH} chunk ${CHUNK_IDX}/${N_CHUNKS} ==="
echo "In:      ${IN_FILE}"
echo "Partial: ${PARTIAL_OUT}"
echo

if [ ! -f "${IN_FILE}" ]; then
    echo "ERROR: input dataset not found: ${IN_FILE}"
    echo "Run parse_month_2200.sh first (MONTH=${MONTH} YEAR=${YEAR})"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/reeval_stockfish.py" \
    --dataset    "${IN_FILE}" \
    --stockfish  "${HOME_DIR}/bin/stockfish" \
    --depth      20 \
    --n          ${N} \
    --seed       ${SEED} \
    --workers    16 \
    --chunk-idx  ${CHUNK_IDX} \
    --n-chunks   ${N_CHUNKS} \
    --partial-out "${PARTIAL_OUT}"

echo
echo "Done. Partial saved: ${PARTIAL_OUT}"
echo "Submit remaining chunks if not done, then run: bsub < jobs/reeval_merge.sh"
