#!/bin/bash
#BSUB -J reeval_merge
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/reeval_merge_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_merge_%J.err

# ── Configuration ──────────────────────────────────────────────────────────────
# Must match the values used in reeval_depth18.sh:
#   bsub -env "MONTH=03,YEAR=2023,DEPTH=18,N_CHUNKS=3,N=300000" < jobs/reeval_merge.sh

MONTH="${MONTH:-03}"
YEAR="${YEAR:-2023}"
DEPTH="${DEPTH:-18}"
N_CHUNKS="${N_CHUNKS:-3}"
N="${N:-300000}"
SEED=42

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}.pt"
OUT_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}_sf${DEPTH}.pt"

# Build the list of partial files
PARTIALS=""
for i in $(seq 0 $((N_CHUNKS - 1))); do
    PART="${BLACKHOLE}/reeval_${YEAR}_${MONTH}_d${DEPTH}_part${i}.pt"
    if [ ! -f "${PART}" ]; then
        echo "ERROR: missing partial file: ${PART}"
        echo "Ensure all ${N_CHUNKS} chunk jobs have completed."
        exit 1
    fi
    PARTIALS="${PARTIALS} ${PART}"
done

echo "=== Merging ${N_CHUNKS} chunks (depth ${DEPTH}) → ${OUT_FILE} ==="
echo "Partials:${PARTIALS}"
echo

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/reeval_stockfish.py" \
    --dataset "${IN_FILE}" \
    --out     "${OUT_FILE}" \
    --merge   ${PARTIALS} \
    --n       ${N} \
    --seed    ${SEED} \
    --min-decisive 0.0 \
    --max-pieces   32

echo
echo "Done. Final dataset: ${OUT_FILE}"
echo
echo "Next — split into decisive/balanced:"
echo "  python3 ${SRC}/split_dataset.py \\"
echo "      --dataset ${OUT_FILE} \\"
echo "      --decisive-out ${HOME_DIR}/Petra-ver2/data/decisive_${YEAR}_${MONTH}.pt \\"
echo "      --balanced-out ${HOME_DIR}/Petra-ver2/data/balanced_${YEAR}_${MONTH}.pt"
