#!/bin/bash
#BSUB -J reeval_drawness_merge
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=24GB]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/reeval_drawness_merge_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_drawness_merge_%J.err

# Merge drawness source SF chunks and derive soft drawness targets from drawn
# games that remain equal under Stockfish and are late enough in the game.

MONTH="${MONTH:-01}"
YEAR="${YEAR:-2025}"
DEPTH="${DEPTH:-18}"
N_CHUNKS="${N_CHUNKS:-12}"
N_CHUNKS_DONE="${N_CHUNKS_DONE:-${N_CHUNKS}}"
N="${N:-3000000}"
SEED="${SEED:-42}"

DRAWNESS_SF_THRESHOLD="${DRAWNESS_SF_THRESHOLD:-0.15}"
DRAWNESS_MIN_PLY="${DRAWNESS_MIN_PLY:-40}"
DRAWNESS_TARGET="${DRAWNESS_TARGET:-0.7}"
DRAWNESS_MAX_PIECES="${DRAWNESS_MAX_PIECES:-32}"

BLACKHOLE="${BLACKHOLE:-/dtu/blackhole/0b/206091}"
HOME_DIR="${HOME_DIR:-/zhome/81/b/206091}"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${IN_FILE:-${BLACKHOLE}/dataset_${YEAR}_${MONTH}_drawsrc.pt}"
OUT_FILE="${OUT_FILE:-${BLACKHOLE}/dataset_${YEAR}_${MONTH}_sf${DEPTH}_drawness.pt}"

PARTIALS=""
for i in $(seq 0 $((N_CHUNKS_DONE - 1))); do
    PART="${BLACKHOLE}/reeval_${YEAR}_${MONTH}_drawness_d${DEPTH}_part${i}.pt"
    if [ ! -f "${PART}" ]; then
        echo "ERROR: missing partial file: ${PART}"
        exit 1
    fi
    PARTIALS="${PARTIALS} ${PART}"
done

echo "=== Merge drawness SF chunks ${YEAR}-${MONTH} depth ${DEPTH} → ${OUT_FILE} ==="
echo "Input:    ${IN_FILE}"
echo "Partials: ${PARTIALS}"
echo "Rule: draw outcome & |SF|<${DRAWNESS_SF_THRESHOLD} & ply>=${DRAWNESS_MIN_PLY} & pieces<=${DRAWNESS_MAX_PIECES} -> target ${DRAWNESS_TARGET}"
echo

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/reeval_stockfish.py" \
    --dataset "${IN_FILE}" \
    --out     "${OUT_FILE}" \
    --merge   ${PARTIALS} \
    --n       "${N}" \
    --seed    "${SEED}" \
    --min-decisive 0.0 \
    --max-pieces   32 \
    --derive-drawness-from-outcome \
    --drawness-sf-threshold "${DRAWNESS_SF_THRESHOLD}" \
    --drawness-min-ply      "${DRAWNESS_MIN_PLY}" \
    --drawness-target       "${DRAWNESS_TARGET}" \
    --drawness-max-pieces   "${DRAWNESS_MAX_PIECES}"

echo
echo "Done. Drawness dataset: ${OUT_FILE}"
