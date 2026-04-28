#!/bin/bash
set -e
#BSUB -J parse_high_elo
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_high_elo_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_high_elo_%J.err

# Parse 250k games at 2200+ ELO from the 2025-02 Lichess database.
# Single job — no chunking needed; the parser scans until 250k qualifying games found.
#
# Pipeline:
#   1. bsub < jobs/parse_high_elo.sh
#   2. bsub -env "CHUNK_IDX=0" < jobs/reeval_high_elo.sh  (× 7, see that script)
#   3. bsub < jobs/reeval_high_elo_merge.sh

MAX_GAMES=250000

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_2025-02.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_high_elo_raw.pt"

echo "=== Parse high-ELO: min_elo=2200  max_games=${MAX_GAMES} ==="
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
    --max-games          ${MAX_GAMES} \
    --min-elo            2200 \
    --positions-per-game 20 \
    --skip-opening       5 \
    --sampling           even \
    --no-strict          \
    --no-split           \
    --checkpoint-every   0

echo
echo "Done. Raw dataset: ${OUT_FILE}"
echo "Check draw rate: python3 src/check_draw_rate.py ${OUT_FILE}"
echo "Next: submit reeval_high_elo.sh (7 chunks)"
