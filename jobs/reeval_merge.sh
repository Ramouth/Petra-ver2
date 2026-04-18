#!/bin/bash
#BSUB -J reeval_merge
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/reeval_merge_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_merge_%J.err

# ── Configuration ──────────────────────────────────────────────────────────────
# Must match the values used in reeval_depth20.sh:
#   MONTH=03 YEAR=2020 bsub < jobs/reeval_merge.sh

MONTH="${MONTH:-03}"
YEAR="${YEAR:-2020}"
N_CHUNKS=6
N=800000
SEED=42

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}.pt"
OUT_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}_sf20.pt"

# Build the list of partial files
PARTIALS=""
for i in $(seq 0 $((N_CHUNKS - 1))); do
    PART="${BLACKHOLE}/reeval_${YEAR}_${MONTH}_d20_part${i}.pt"
    if [ ! -f "${PART}" ]; then
        echo "ERROR: missing partial file: ${PART}"
        echo "Ensure all ${N_CHUNKS} chunk jobs have completed."
        exit 1
    fi
    PARTIALS="${PARTIALS} ${PART}"
done

echo "=== Merging ${N_CHUNKS} chunks → ${OUT_FILE} ==="
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
