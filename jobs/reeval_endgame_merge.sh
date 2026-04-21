#!/bin/bash
#BSUB -J reeval_endgame_merge
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/reeval_endgame_merge_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_endgame_merge_%J.err

# Merge 4 SF depth-15 endgame re-evaluation chunks → endgame_sf15.pt.
# Run after all 4 reeval_endgame_sf15.sh chunks complete.

N=1000000
N_CHUNKS=4
DEPTH=15
SEED=42

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/endgame_raw.pt"
OUT_FILE="${BLACKHOLE}/endgame_sf15.pt"

PARTIALS=""
for i in $(seq 0 $((N_CHUNKS - 1))); do
    PART="${BLACKHOLE}/endgame_sf15_part${i}.pt"
    if [ ! -f "${PART}" ]; then
        echo "ERROR: missing partial file: ${PART}"
        echo "Ensure all ${N_CHUNKS} reeval_endgame_sf15 chunks have completed."
        exit 1
    fi
    PARTIALS="${PARTIALS} ${PART}"
done

echo "=== Merging ${N_CHUNKS} endgame chunks (depth ${DEPTH}) → ${OUT_FILE} ==="
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
echo "Done. Endgame pool: ${OUT_FILE}"
echo
echo "This file is ready to use as --endgame-dataset in train.py."
echo "Recommended anchor fraction: 0.25-0.35 (200k-280k positions per epoch)"
