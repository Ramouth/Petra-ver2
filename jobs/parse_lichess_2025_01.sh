#!/bin/bash
#BSUB -J parse_2025_01
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_2025_01_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_2025_01_%J.err

# Parse Lichess 2025-01 (31GB compressed, ~150M games).
#
# ELO floor 1850: gives a healthy mix of decisive and balanced positions across
# all game phases. Lower ELO adds blunders (better SF signal range); higher ELO
# compresses labels near zero (geometry regression observed with 2500+ ELO).
# 2025-01 has abundant games at 1850+ — 150k qualifying games is easily reachable.
#
# Output: dataset_2025_01.pt (~1.5M raw positions before SF reeval)
# Next:   bsub -env "MONTH=01,YEAR=2025,CHUNK_IDX=0" < jobs/reeval_depth18.sh
#         bsub -env "MONTH=01,YEAR=2025,CHUNK_IDX=1" < jobs/reeval_depth18.sh
#         bsub -env "MONTH=01,YEAR=2025,CHUNK_IDX=2" < jobs/reeval_depth18.sh
#         bsub -env "MONTH=01,YEAR=2025,CHUNK_IDX=3" < jobs/reeval_depth18.sh
#         bsub -env "MONTH=01,YEAR=2025" < jobs/reeval_merge.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_2025-01.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_2025_01.pt"

echo "=== Parsing Lichess 2025-01 (min_elo=1850, max_games=150000) ==="
echo "PGN:  ${PGN_FILE}"
echo "Out:  ${OUT_FILE}"
echo

if [ ! -f "${PGN_FILE}" ]; then
    echo "ERROR: PGN not found at ${PGN_FILE}"
    echo "Download with:"
    echo "  wget -P ${BLACKHOLE} https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
pip install zstandard --quiet

python3 -u "${SRC}/data.py" \
    --pgn                "${PGN_FILE}" \
    --out                "${OUT_FILE}" \
    --max-games          150000 \
    --min-elo            1850 \
    --positions-per-game 20 \
    --skip-opening       5 \
    --sampling           even \
    --no-strict          \
    --checkpoint-every   10000

echo
echo "Done. Output: ${OUT_FILE}"
echo
echo "Next — submit 4 SF reeval chunks:"
echo "  bsub -env \"MONTH=01,YEAR=2025,CHUNK_IDX=0\" < jobs/reeval_depth18.sh"
echo "  bsub -env \"MONTH=01,YEAR=2025,CHUNK_IDX=1\" < jobs/reeval_depth18.sh"
echo "  bsub -env \"MONTH=01,YEAR=2025,CHUNK_IDX=2\" < jobs/reeval_depth18.sh"
echo "  bsub -env \"MONTH=01,YEAR=2025,CHUNK_IDX=3\" < jobs/reeval_depth18.sh"
