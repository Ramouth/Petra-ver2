#!/bin/bash
#BSUB -J parse_phase15_mid
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_phase15_mid_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_phase15_mid_%J.err

# Phase 1.5 mid-band parse: 1800-2400 ELO, 100k games.
# Expands Phase 1's 1800-2300 cutoff to 2400 — 2300-2400 games are
# technically clean without the label-collapse seen at 2500+.
# Larger game count (100k vs 33k) gives more position diversity.
#
#   bsub < jobs/parse_phase15_mid.sh
#
# Next: submit 3 reeval chunks in parallel:
#   bsub -env "CHUNK_IDX=0" < jobs/reeval_phase15_mid.sh
#   bsub -env "CHUNK_IDX=1" < jobs/reeval_phase15_mid.sh
#   bsub -env "CHUNK_IDX=2" < jobs/reeval_phase15_mid.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_2021-06.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_phase15_mid.pt"

echo "=== Phase 1.5 mid parse: 2021-06, ELO 1800-2400, 100k games ==="
echo "PGN: ${PGN_FILE}"
echo "Out: ${OUT_FILE}"
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
    --max-games          100000 \
    --min-elo            1800 \
    --max-elo            2400 \
    --positions-per-game 20 \
    --skip-opening       5 \
    --sampling           even \
    --no-strict          \
    --checkpoint-every   10000

echo
echo "Done. Output: ${OUT_FILE}"
echo "Next (submit all 3 in parallel):"
echo "  bsub -env \"CHUNK_IDX=0\" < jobs/reeval_phase15_mid.sh"
echo "  bsub -env \"CHUNK_IDX=1\" < jobs/reeval_phase15_mid.sh"
echo "  bsub -env \"CHUNK_IDX=2\" < jobs/reeval_phase15_mid.sh"
