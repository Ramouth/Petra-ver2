#!/bin/bash
#BSUB -J parse_drawness_source
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=24GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_drawness_source_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_drawness_source_%J.err

# Re-parse PGN with metadata needed for drawn-game soft drawness:
#   outcome_values, game_ids, plys.
#
# This is required for old SF-labelled datasets that lack the original game
# outcome and ply metadata. The resulting raw dataset must then be passed through
# reeval_drawness_depth18.sh + reeval_drawness_merge.sh.
#
# Override example:
#   bsub -env "YEAR=2025,MONTH=01,PGN_FILE=/dtu/blackhole/0b/206091/lichess_db_standard_rated_2025-01.pgn.zst,MAX_GAMES=150000" < jobs/parse_drawness_source.sh

MONTH="${MONTH:-01}"
YEAR="${YEAR:-2025}"
MAX_GAMES="${MAX_GAMES:-150000}"
MIN_ELO="${MIN_ELO:-1850}"
POSITIONS_PER_GAME="${POSITIONS_PER_GAME:-20}"
SKIP_OPENING="${SKIP_OPENING:-5}"
SAMPLING="${SAMPLING:-even}"

BLACKHOLE="${BLACKHOLE:-/dtu/blackhole/0b/206091}"
HOME_DIR="${HOME_DIR:-/zhome/81/b/206091}"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${PGN_FILE:-${BLACKHOLE}/lichess_db_standard_rated_${YEAR}-${MONTH}.pgn.zst}"
OUT_FILE="${OUT_FILE:-${BLACKHOLE}/dataset_${YEAR}_${MONTH}_drawsrc.pt}"

echo "=== Parse drawness source dataset ${YEAR}-${MONTH} ==="
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
    --max-games          "${MAX_GAMES}" \
    --min-elo            "${MIN_ELO}" \
    --positions-per-game "${POSITIONS_PER_GAME}" \
    --skip-opening       "${SKIP_OPENING}" \
    --sampling           "${SAMPLING}" \
    --no-strict          \
    --checkpoint-every   10000

echo
echo "Done. Raw drawness source: ${OUT_FILE}"
echo
echo "Next submit SF re-eval chunks, for example:"
echo "  bsub -env \"YEAR=${YEAR},MONTH=${MONTH},CHUNK_IDX=0\" < jobs/reeval_drawness_depth18.sh"
