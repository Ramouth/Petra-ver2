#!/bin/bash
#BSUB -J parse_month
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_month_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_month_%J.err

# ── Configuration ──────────────────────────────────────────────────────────────
MONTH="${MONTH:-03}"
YEAR="${YEAR:-2023}"

# ELO filter — scales with dataset size:
#   2020 (~15M games/month):  2200+ gives ~30-80k qualifying games
#   2023 (~108M games/month): 2500+ gives ~100k+ qualifying games — use the abundance
MIN_ELO="${MIN_ELO:-2500}"

# How many qualifying games to keep.
# With 108M games and min_elo=2500, stopping at 300k games is a small fraction
# of what's available — no need to scan the whole file.
MAX_GAMES="${MAX_GAMES:-100000}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_${YEAR}-${MONTH}.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}.pt"

# ── Wall time note ─────────────────────────────────────────────────────────────
# March 2023 PGN is large (~50-100GB compressed). The parser scans sequentially
# and stops once MAX_GAMES qualifying games are found. At 2500+ ELO, qualifying
# games are ~0.1-0.5% of the file, so expect to scan ~60-300M games to collect
# 300k. Empirically this takes 4-10h — set wall to 12h to be safe.

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
echo "  MONTH=${MONTH} YEAR=${YEAR} CHUNK_IDX=0 bsub < jobs/reeval_depth20.sh"
echo "  MONTH=${MONTH} YEAR=${YEAR} CHUNK_IDX=1 bsub < jobs/reeval_depth20.sh"
echo "  MONTH=${MONTH} YEAR=${YEAR} CHUNK_IDX=2 bsub < jobs/reeval_depth20.sh"
