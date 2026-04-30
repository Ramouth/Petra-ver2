#!/bin/bash
set -euo pipefail
#BSUB -J parse_2025_04_2400
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_2025_04_2400_c%I_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_2025_04_2400_c%I_%J.err

# Parse Lichess 2025-04 at min_elo=2400, ONE OF THREE CHUNKS.
# Sister to parse_2025_02_2400.sh — independent month, can run in parallel
# with both its own chunks and 2025-02's chunks (6 jobs total).
#
#   bsub -env "CHUNK_IDX=0" < jobs/parse_2025_04_2400.sh
#   bsub -env "CHUNK_IDX=1" < jobs/parse_2025_04_2400.sh
#   bsub -env "CHUNK_IDX=2" < jobs/parse_2025_04_2400.sh

YEAR=2025
MONTH=04
MIN_ELO=2400
GAMES_PER_CHUNK=100000

CHUNK_IDX="${CHUNK_IDX:?CHUNK_IDX must be set (0, 1, or 2)}"
SKIP_GAMES=$(( CHUNK_IDX * GAMES_PER_CHUNK ))

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_${YEAR}-${MONTH}.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}_chunk${CHUNK_IDX}.pt"

echo "=== Parsing Lichess ${YEAR}-${MONTH} chunk ${CHUNK_IDX} (skip=${SKIP_GAMES}, take=${GAMES_PER_CHUNK}, min_elo=${MIN_ELO}) ==="
echo "PGN:  ${PGN_FILE}"
echo "Out:  ${OUT_FILE}"
echo

if [ ! -f "${PGN_FILE}" ]; then
    echo "ERROR: PGN not found at ${PGN_FILE}"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
pip install zstandard --quiet

python3 -u "${SRC}/data.py" \
    --pgn                "${PGN_FILE}" \
    --out                "${OUT_FILE}" \
    --skip-games         ${SKIP_GAMES} \
    --max-games          ${GAMES_PER_CHUNK} \
    --min-elo            ${MIN_ELO} \
    --positions-per-game 20 \
    --skip-opening       5 \
    --sampling           even \
    --no-split           \
    --no-strict          \
    --checkpoint-every   10000

echo
echo "Done. Chunk: ${OUT_FILE}"
echo "After all 3 chunks finish, merge with: bsub -env \"MONTH=${MONTH}\" < jobs/merge_2025_chunks.sh"
