#!/bin/bash
#BSUB -J reeval_depth18
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 12:00
#BSUB -o /zhome/81/b/206091/logs/reeval_depth18_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_depth18_%J.err

# ── Configuration ──────────────────────────────────────────────────────────────
# Set MONTH, YEAR, and CHUNK_IDX before submitting.
# Use -env to pass variables — plain "VAR=val bsub < script" is unreliable on
# some LSF configurations and the job will silently fall back to defaults.
#
#   bsub -env "MONTH=03,YEAR=2023,CHUNK_IDX=0" < jobs/reeval_depth18.sh
#   bsub -env "MONTH=03,YEAR=2023,CHUNK_IDX=1" < jobs/reeval_depth18.sh
#   bsub -env "MONTH=03,YEAR=2023,CHUNK_IDX=2" < jobs/reeval_depth18.sh
#   bsub -env "MONTH=03,YEAR=2023,CHUNK_IDX=3" < jobs/reeval_depth18.sh
#
# After all 4 finish, run the merge job:
#   bsub -env "MONTH=03,YEAR=2023" < jobs/reeval_merge.sh
#
# If a chunk hits the wall, it saves a checkpoint with however far it got.
# The merge will report which positions are missing — resubmit that CHUNK_IDX.

MONTH="${MONTH:-03}"
YEAR="${YEAR:-2023}"
CHUNK_IDX="${CHUNK_IDX:-0}"
N_CHUNKS=4          # 800k positions / 4 = 200k per chunk → ~6-8h at depth 18
N=800000
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
