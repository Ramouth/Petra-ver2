#!/bin/bash
#BSUB -J reeval_depth18
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB] select[hname!='n-62-21-89']"
#BSUB -W 12:00
#BSUB -o /zhome/81/b/206091/logs/reeval_depth18_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_depth18_%J.err

# ── Configuration ──────────────────────────────────────────────────────────────
# Iterative reeval: 3M positions split into 12 chunks of 250k, submitted in
# rounds of 4. Train after each round, then submit the next 4 chunks.
#
# Round 1 (chunks 0-3):
#   bsub -env "MONTH=01,YEAR=2025,CHUNK_IDX=0" < jobs/reeval_depth18.sh
#   bsub -env "MONTH=01,YEAR=2025,CHUNK_IDX=1" < jobs/reeval_depth18.sh
#   bsub -env "MONTH=01,YEAR=2025,CHUNK_IDX=2" < jobs/reeval_depth18.sh
#   bsub -env "MONTH=01,YEAR=2025,CHUNK_IDX=3" < jobs/reeval_depth18.sh
#   bsub -env "MONTH=01,YEAR=2025,N_CHUNKS_DONE=4,ROUND=1" < jobs/reeval_merge.sh
#
# Round 2 (chunks 4-7) — after training on round 1:
#   bsub -env "MONTH=01,YEAR=2025,CHUNK_IDX=4" < jobs/reeval_depth18.sh
#   ...CHUNK_IDX=5,6,7...
#   bsub -env "MONTH=01,YEAR=2025,N_CHUNKS_DONE=8,ROUND=2" < jobs/reeval_merge.sh
#
# Round 3 (chunks 8-11):
#   bsub -env "MONTH=01,YEAR=2025,CHUNK_IDX=8" < jobs/reeval_depth18.sh
#   ...CHUNK_IDX=9,10,11...
#   bsub -env "MONTH=01,YEAR=2025,N_CHUNKS_DONE=12,ROUND=3" < jobs/reeval_merge.sh
#
# If a chunk hits the wall, it saves a checkpoint. Resubmit that CHUNK_IDX.

MONTH="${MONTH:-03}"
YEAR="${YEAR:-2023}"
CHUNK_IDX="${CHUNK_IDX:-0}"
N_CHUNKS="${N_CHUNKS:-12}"
N="${N:-3000000}"
SEED=42

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}.pt"
PARTIAL_OUT="${BLACKHOLE}/reeval_${YEAR}_${MONTH}_d18_part${CHUNK_IDX}.pt"

echo "=== SF depth-18 re-evaluation: ${YEAR}-${MONTH} chunk ${CHUNK_IDX}/${N_CHUNKS} ==="
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
    --depth      18 \
    --n          ${N} \
    --seed       ${SEED} \
    --workers    16 \
    --chunk-idx  ${CHUNK_IDX} \
    --n-chunks   ${N_CHUNKS} \
    --partial-out "${PARTIAL_OUT}"

echo
echo "Done. Partial saved: ${PARTIAL_OUT}"
echo "Submit remaining chunks if not done, then run: bsub < jobs/reeval_merge.sh"
