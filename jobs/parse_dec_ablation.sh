#!/bin/bash
#BSUB -J parse_dec_c${CHUNK_IDX:-0}
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_dec_ablation_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_dec_ablation_%J.err

# Decisiveness ablation — parallel parse. 5 chunks × 50k games = 250k total.
# Each chunk skips N games and reads the next 50k, writing a raw (no-split) .pt.
# Submit all 5 in parallel, then run parse_dec_ablation_merge.sh.
#
#   bsub -env "CHUNK_IDX=0" < jobs/parse_dec_ablation.sh
#   bsub -env "CHUNK_IDX=1" < jobs/parse_dec_ablation.sh
#   bsub -env "CHUNK_IDX=2" < jobs/parse_dec_ablation.sh
#   bsub -env "CHUNK_IDX=3" < jobs/parse_dec_ablation.sh
#   bsub -env "CHUNK_IDX=4" < jobs/parse_dec_ablation.sh
#
# When all 5 done: bsub < jobs/parse_dec_ablation_merge.sh

CHUNK_IDX="${CHUNK_IDX:-0}"
GAMES_PER_CHUNK=50000

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_2025-02.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_dec_ablation_raw_c${CHUNK_IDX}.pt"
SKIP=$((CHUNK_IDX * GAMES_PER_CHUNK))

echo "=== Parse dec_ablation chunk ${CHUNK_IDX}: skip=${SKIP}, games=${GAMES_PER_CHUNK} ==="
echo "PGN: ${PGN_FILE}"
echo "Out: ${OUT_FILE}"
echo

if [ ! -f "${PGN_FILE}" ]; then
    echo "ERROR: PGN not found: ${PGN_FILE}"
    echo "Download with:"
    echo "  wget -P ${BLACKHOLE} https://database.lichess.org/standard/lichess_db_standard_rated_2025-02.pgn.zst"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
pip install zstandard --quiet

python3 -u "${SRC}/data.py" \
    --pgn                "${PGN_FILE}" \
    --out                "${OUT_FILE}" \
    --max-games          ${GAMES_PER_CHUNK} \
    --skip-games         ${SKIP} \
    --min-elo            1800 \
    --max-elo            2450 \
    --positions-per-game 20 \
    --skip-opening       5 \
    --sampling           even \
    --no-strict          \
    --no-split           \
    --checkpoint-every   0

echo
echo "Done. Raw chunk: ${OUT_FILE}"
