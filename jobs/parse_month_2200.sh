#!/bin/bash
#BSUB -J parse_month
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_month_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_month_%J.err

# ── Configuration ──────────────────────────────────────────────────────────────
MONTH="${MONTH:-03}"
YEAR="${YEAR:-2023}"

# ELO filter — lower = more decisive positions, better value range coverage.
#   2500+ ELO games are too balanced: SF labels cluster near 0, compressing
#   the geometry and hurting value calibration (see lichess_2023_03 regression).
#   1850 is the target: enough decisive games to give geometry signal while
#   retaining draws and complex middlegame positions for breadth (slow staircase).
#   Hard-filtering by piece count or decisive-only collapses the complexity
#   that geometry also needs — 1850 gives the combination naturally.
#   2023 (~108M games/month): 1850+ gives tens of millions — abundance fine.
MIN_ELO="${MIN_ELO:-1850}"

# How many qualifying games to keep.
# With 108M games and min_elo=2000, qualifying games are plentiful — 100k is fine.
MAX_GAMES="${MAX_GAMES:-100000}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_${YEAR}-${MONTH}.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}.pt"

# ── Wall time note ─────────────────────────────────────────────────────────────
# March 2023 PGN is large (~50-100GB compressed). The parser scans sequentially
# and stops once MAX_GAMES qualifying games are found. At 2000+ ELO, qualifying
# games are ~1-5% of the file — much faster to find 100k than at 2500+.
# Empirically 2-4h — 24h wall is conservative.

echo "=== Parsing Lichess ${YEAR}-${MONTH} (min_elo=${MIN_ELO}, max_games=${MAX_GAMES}) ==="
echo "PGN:  ${PGN_FILE}"
echo "Out:  ${OUT_FILE}"
echo

if [ ! -f "${PGN_FILE}" ]; then
    echo "ERROR: PGN not found at ${PGN_FILE}"
    echo "Download with:"
    echo "  wget -P ${BLACKHOLE} https://database.lichess.org/standard/lichess_db_standard_rated_${YEAR}-${MONTH}.pgn.zst"
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
echo "Next step: submit 3 chunk jobs:"
echo "  MONTH=${MONTH} YEAR=${YEAR} CHUNK_IDX=0 bsub < jobs/reeval_depth18.sh"
echo "  MONTH=${MONTH} YEAR=${YEAR} CHUNK_IDX=1 bsub < jobs/reeval_depth18.sh"
echo "  MONTH=${MONTH} YEAR=${YEAR} CHUNK_IDX=2 bsub < jobs/reeval_depth18.sh"
