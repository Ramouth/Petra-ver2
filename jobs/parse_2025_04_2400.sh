#!/bin/bash
set -euo pipefail
#BSUB -J parse_2025_04_2400
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_2025_04_2400_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_2025_04_2400_%J.err

# Parse Lichess 2025-04 at min_elo=2400 — fresh source for the `big` model.
# 2025-04 is untouched in blackhole, so no overlap with any existing model's
# training data. min_elo=2400 ≈ FIDE 2100+ — strong amateur play, ~7-10%
# natural draw rate (matches the target draw fraction for big).
#
# 300k qualifying games × 20 positions/game = ~6M raw positions, plenty of
# headroom for build_big to engineer 5M @ 7.5% drawn after dedup.
#
#   bsub < jobs/parse_2025_04_2400.sh

YEAR=2025
MONTH=04
MIN_ELO=2400
MAX_GAMES=300000

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_${YEAR}-${MONTH}.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}.pt"

echo "=== Parsing Lichess ${YEAR}-${MONTH} (min_elo=${MIN_ELO}, max_games=${MAX_GAMES}) ==="
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
    --max-games          ${MAX_GAMES} \
    --min-elo            ${MIN_ELO} \
    --positions-per-game 20 \
    --skip-opening       5 \
    --sampling           even \
    --no-strict          \
    --checkpoint-every   10000

echo
echo "Done. Output: ${OUT_FILE}"
echo "Next: bsub < jobs/build_big.sh"
