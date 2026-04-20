#!/bin/bash
#BSUB -J parse_decisive_endgames
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_decisive_endgames_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_decisive_endgames_%J.err

# ── Configuration ──────────────────────────────────────────────────────────────
MONTH="${MONTH:-03}"
YEAR="${YEAR:-2023}"

# Endgame bootstrap dataset: decisive games only, endgame positions only.
#
# Rationale: geometry rank needs decisive SF signals (|v| >> 0) to separate
# win/draw/loss centroids. Opening/middlegame positions at 2000+ ELO are nearly
# equal — SF labels cluster near 0 → compressed geometry.
#
# Strategy: filter to positions with ≤ MAX_PIECES pieces (endgame) from games
# with a decisive result (no draws). SF will evaluate these positions strongly,
# giving the rank-reg training decisive signals to build geometry on.
# Complexity (broader positions) follows in subsequent training stages once
# geometry is bootstrapped.
#
# MAX_PIECES=16: typical endgame (≤2 rooks, pawns, kings).
# MAX_PIECES=20: includes rook+minor endgames with some pawns.
MAX_PIECES="${MAX_PIECES:-16}"

# ELO filter: lower is fine here — we're filtering by decisiveness, not ELO.
# More games → more endgame positions to sample from.
MIN_ELO="${MIN_ELO:-1800}"

# Positions per game: high because many positions will be filtered by MAX_PIECES.
# Most games produce only 5-20 qualifying endgame positions.
POSITIONS_PER_GAME="${POSITIONS_PER_GAME:-30}"

# How many qualifying games to scan (not all will yield endgame positions).
MAX_GAMES="${MAX_GAMES:-500000}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_${YEAR}-${MONTH}.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_decisive_endgames_${YEAR}_${MONTH}.pt"

echo "=== Decisive endgames parse: ${YEAR}-${MONTH} ==="
echo "  max_pieces=${MAX_PIECES}, min_elo=${MIN_ELO}, decisive_only=true"
echo "  max_games=${MAX_GAMES}, positions_per_game=${POSITIONS_PER_GAME}"
echo "PGN: ${PGN_FILE}"
echo "Out: ${OUT_FILE}"
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
    --positions-per-game ${POSITIONS_PER_GAME} \
    --skip-opening       5 \
    --sampling           even \
    --no-strict          \
    --decisive-only      \
    --max-pieces         ${MAX_PIECES} \
    --checkpoint-every   10000

echo
echo "Done. Output: ${OUT_FILE}"
echo "Next: submit 3 reeval chunks (depth 18):"
echo "  MONTH=${MONTH} YEAR=${YEAR} DATASET=decisive_endgames bsub < jobs/reeval_depth18.sh"
