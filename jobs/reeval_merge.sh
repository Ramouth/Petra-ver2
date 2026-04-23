#!/bin/bash
#BSUB -J reeval_merge
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/reeval_merge_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_merge_%J.err

# ── Configuration ──────────────────────────────────────────────────────────────
# N_CHUNKS_DONE controls how many partial files to merge (for iterative rounds).
# ROUND is appended to the output filename so each round produces a distinct file.
#
#   bsub -env "MONTH=01,YEAR=2025,N_CHUNKS_DONE=4,ROUND=1"  < jobs/reeval_merge.sh
#   bsub -env "MONTH=01,YEAR=2025,N_CHUNKS_DONE=8,ROUND=2"  < jobs/reeval_merge.sh
#   bsub -env "MONTH=01,YEAR=2025,N_CHUNKS_DONE=12,ROUND=3" < jobs/reeval_merge.sh

MONTH="${MONTH:-03}"
YEAR="${YEAR:-2023}"
DEPTH="${DEPTH:-18}"
N_CHUNKS="${N_CHUNKS:-12}"
N_CHUNKS_DONE="${N_CHUNKS_DONE:-${N_CHUNKS}}"
N="${N:-3000000}"
ROUND="${ROUND:-1}"
SEED=42

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}.pt"
OUT_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}_sf${DEPTH}_r${ROUND}.pt"

# Build the list of partial files (chunks 0 .. N_CHUNKS_DONE-1)
PARTIALS=""
for i in $(seq 0 $((N_CHUNKS_DONE - 1))); do
    PART="${BLACKHOLE}/reeval_${YEAR}_${MONTH}_d${DEPTH}_part${i}.pt"
    if [ ! -f "${PART}" ]; then
        echo "ERROR: missing partial file: ${PART}"
        echo "Ensure chunks 0-$((N_CHUNKS_DONE - 1)) have completed."
        exit 1
    fi
    PARTIALS="${PARTIALS} ${PART}"
done

echo "=== Merging chunks 0-$((N_CHUNKS_DONE-1)) of ${N_CHUNKS} (depth ${DEPTH}, round ${ROUND}) → ${OUT_FILE} ==="
echo "Partials:${PARTIALS}"
echo

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/reeval_stockfish.py" \
    --dataset                      "${IN_FILE}" \
    --out                          "${OUT_FILE}" \
    --merge                        ${PARTIALS} \
    --n                            ${N} \
    --seed                         ${SEED} \
    --min-decisive                 0.0 \
    --max-pieces                   32 \
    --derive-drawness-from-outcome \
    --drawness-game-level \
    --drawness-sf-threshold        0.11 \
    --drawness-target              0.8 \
    --allow-partial-coverage

echo
echo "Done. Final dataset: ${OUT_FILE}"
echo
echo "Next — split into decisive/balanced:"
echo "  python3 ${SRC}/split_dataset.py \\"
echo "      --dataset ${OUT_FILE} \\"
echo "      --decisive-out ${HOME_DIR}/Petra-ver2/data/decisive_${YEAR}_${MONTH}.pt \\"
echo "      --balanced-out ${HOME_DIR}/Petra-ver2/data/balanced_${YEAR}_${MONTH}.pt"
