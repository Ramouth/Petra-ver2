#!/bin/bash
#BSUB -J reeval_endgame_sf15
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=4GB] select[hname!='n-62-21-89']"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/reeval_endgame_sf15_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_endgame_sf15_%J.err

# SF depth-15 re-evaluation of endgame_raw.pt (250k positions per chunk).
#
# Submit all 4 chunks simultaneously after gen_endgame_raw.sh finishes:
#   bsub -env "CHUNK_IDX=0" < jobs/reeval_endgame_sf15.sh
#   bsub -env "CHUNK_IDX=1" < jobs/reeval_endgame_sf15.sh
#   bsub -env "CHUNK_IDX=2" < jobs/reeval_endgame_sf15.sh
#   bsub -env "CHUNK_IDX=3" < jobs/reeval_endgame_sf15.sh
#
# After all 4 finish, run:
#   bsub < jobs/reeval_endgame_merge.sh
#
# Depth 15 is sufficient for endgame positions (3-6 pieces);
# endgames evaluate ~3-5x faster than depth-18 middlegames.
# 250k / ~40 pos/s ≈ 1.75h → 4h wall time is safe.

CHUNK_IDX="${CHUNK_IDX:-0}"
N_CHUNKS=4
N=1000000
DEPTH=15
SEED=42

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/endgame_raw.pt"
PARTIAL_OUT="${BLACKHOLE}/endgame_sf15_part${CHUNK_IDX}.pt"

echo "=== SF depth-15 re-eval: endgame chunk ${CHUNK_IDX}/${N_CHUNKS} ==="
echo "In:      ${IN_FILE}"
echo "Partial: ${PARTIAL_OUT}"
echo

if [ ! -f "${IN_FILE}" ]; then
    echo "ERROR: input dataset not found: ${IN_FILE}"
    echo "Run gen_endgame_raw.sh first."
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/reeval_stockfish.py" \
    --dataset     "${IN_FILE}" \
    --stockfish   "${HOME_DIR}/bin/stockfish" \
    --depth       ${DEPTH} \
    --n           ${N} \
    --seed        ${SEED} \
    --workers     16 \
    --chunk-idx   ${CHUNK_IDX} \
    --n-chunks    ${N_CHUNKS} \
    --partial-out "${PARTIAL_OUT}"

echo
echo "Done. Partial saved: ${PARTIAL_OUT}"
echo "Submit remaining chunks if not done, then run: bsub < jobs/reeval_endgame_merge.sh"
