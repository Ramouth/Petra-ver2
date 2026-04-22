#!/bin/bash
#BSUB -J reeval_drawness_d18
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB] select[hname!='n-62-21-89']"
#BSUB -W 12:00
#BSUB -o /zhome/81/b/206091/logs/reeval_drawness_d18_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_drawness_d18_%J.err

# Stockfish re-evaluation chunks for the metadata-preserving drawness source
# dataset. The drawness targets are derived during the merge step, after all SF
# values are available.

MONTH="${MONTH:-01}"
YEAR="${YEAR:-2025}"
CHUNK_IDX="${CHUNK_IDX:-0}"
N_CHUNKS="${N_CHUNKS:-12}"
N="${N:-3000000}"
SEED="${SEED:-42}"
DEPTH="${DEPTH:-18}"

BLACKHOLE="${BLACKHOLE:-/dtu/blackhole/0b/206091}"
HOME_DIR="${HOME_DIR:-/zhome/81/b/206091}"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${IN_FILE:-${BLACKHOLE}/dataset_${YEAR}_${MONTH}_drawsrc.pt}"
PARTIAL_OUT="${PARTIAL_OUT:-${BLACKHOLE}/reeval_${YEAR}_${MONTH}_drawness_d${DEPTH}_part${CHUNK_IDX}.pt}"

echo "=== Drawness source SF depth-${DEPTH}: ${YEAR}-${MONTH} chunk ${CHUNK_IDX}/${N_CHUNKS} ==="
echo "In:      ${IN_FILE}"
echo "Partial: ${PARTIAL_OUT}"
echo

if [ ! -f "${IN_FILE}" ]; then
    echo "ERROR: input dataset not found: ${IN_FILE}"
    echo "Run jobs/parse_drawness_source.sh first."
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/reeval_stockfish.py" \
    --dataset     "${IN_FILE}" \
    --stockfish   "${HOME_DIR}/bin/stockfish" \
    --depth       "${DEPTH}" \
    --n           "${N}" \
    --seed        "${SEED}" \
    --workers     16 \
    --chunk-idx   "${CHUNK_IDX}" \
    --n-chunks    "${N_CHUNKS}" \
    --partial-out "${PARTIAL_OUT}"

echo
echo "Done. Partial saved: ${PARTIAL_OUT}"
