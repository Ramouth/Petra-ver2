#!/bin/bash
#BSUB -J parse_low_elo_2025_01
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_low_elo_2025_01_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_low_elo_2025_01_%J.err

# Parse Lichess 2025-01 in the 1600-1850 ELO band.
#
# Motivation: lower-rated games contain more varied openings, unconventional
# piece placements, and fewer theory repetitions. This forces the geometry
# to span a wider manifold, increasing effective rank beyond what 1850+ games
# alone can provide.
#
# ELO band 1600-1850:
#   - Above pure beginner noise (1200-) that lacks meaningful structure
#   - Below the theory-convergent zone (1850+) we already have
#   - Dense population on Lichess → 100k qualifying games are plentiful
#
# After this job:
#   bsub -env "YEAR=2025,MONTH=01,SUFFIX=low_elo" < jobs/reeval_low_elo.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_2025-01.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_2025_01_low_elo.pt"

echo "=== Parsing Lichess 2025-01 (ELO 1600-1850) ==="
echo "PGN: ${PGN_FILE}"
echo "Out: ${OUT_FILE}"
echo

if [ ! -f "${PGN_FILE}" ]; then
    echo "ERROR: PGN not found: ${PGN_FILE}"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
pip install zstandard --quiet

python3 -u "${SRC}/data.py" \
    --pgn                "${PGN_FILE}" \
    --out                "${OUT_FILE}" \
    --max-games          100000 \
    --min-elo            1600 \
    --max-elo            1850 \
    --positions-per-game 20 \
    --skip-opening       5 \
    --sampling           even \
    --no-strict \
    --checkpoint-every   10000

echo
echo "Done. Output: ${OUT_FILE}"
echo "Next: bsub -env \"SUFFIX=low_elo\" < jobs/reeval_low_elo.sh"
