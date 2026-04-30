#!/bin/bash
set -euo pipefail
#BSUB -J parse_chunk_5h
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 5:00
#BSUB -o /zhome/81/b/206091/logs/parse_chunk_5h_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_chunk_5h_%J.err

# Conservative single-chunk parse: tight 5-hour wall + stall watchdog
# (baked into data.py: warn @ 10 min idle, hard exit @ 20 min idle).
#
# Empirical rate at min_elo=2400 on 2025 PGNs: ~5.15 games/sec.
# 80k games ≈ 4.3h actual parse + ~10 min env/fast-forward overhead → fits in 5h.
# data.py still ckpts every 5k games, so a wall-time SIGTERM saves all but
# the most recent <5k games.
#
# Output goes to *_5h.pt to avoid clobbering existing partial ckpts on blackhole.
#
# Submit:
#   bsub -env "YEAR=2025,MONTH=04,CHUNK_IDX=0" < jobs/parse_chunk_5h.sh
#   bsub -env "YEAR=2025,MONTH=02,CHUNK_IDX=1,MIN_ELO=2400,MAX_GAMES=80000" \
#        < jobs/parse_chunk_5h.sh
#
# Defaults: MIN_ELO=2400, MAX_GAMES=80000.

YEAR="${YEAR:?YEAR must be set (e.g. 2025)}"
MONTH="${MONTH:?MONTH must be set, two digits (e.g. 04)}"
CHUNK_IDX="${CHUNK_IDX:?CHUNK_IDX must be set (0, 1, ...)}"
MIN_ELO="${MIN_ELO:-2400}"
MAX_GAMES="${MAX_GAMES:-80000}"

SKIP_GAMES=$(( CHUNK_IDX * MAX_GAMES ))

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_${YEAR}-${MONTH}.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}_chunk${CHUNK_IDX}_5h.pt"

echo "=== parse_chunk_5h: ${YEAR}-${MONTH} chunk ${CHUNK_IDX} ==="
echo "  skip_games  = ${SKIP_GAMES}"
echo "  max_games   = ${MAX_GAMES}"
echo "  min_elo     = ${MIN_ELO}"
echo "  PGN         = ${PGN_FILE}"
echo "  Out         = ${OUT_FILE}"
echo "  Wall        = 5h"
echo "  Watchdog    = warn 10 min idle / hard exit 20 min idle (default)"
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
    --max-games          ${MAX_GAMES} \
    --min-elo            ${MIN_ELO} \
    --positions-per-game 20 \
    --skip-opening       5 \
    --sampling           even \
    --no-split           \
    --no-strict          \
    --checkpoint-every   5000

echo
echo "Done. Chunk: ${OUT_FILE}"
echo "  Final .pt was written if parsing reached --max-games."
echo "  If wall/watchdog killed the job, only ${OUT_FILE%.pt}.ckpt.pt is on disk;"
echo "  finalize it with:"
echo "    python3 ${SRC}/data.py --from-checkpoint ${OUT_FILE%.pt}.ckpt.pt --out ${OUT_FILE} --no-split --no-strict"
