#!/bin/bash
set -e
#BSUB -J parse_elo_ablation
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_elo_ablation_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_elo_ablation_%J.err

# ELO ablation parse — scans 2025-02 PGN until 250k qualifying games found.
# All three can run in parallel.
#
#   bsub -env "MIN_ELO=2000" < jobs/parse_elo_ablation.sh
#   bsub -env "MIN_ELO=2100" < jobs/parse_elo_ablation.sh
#   bsub -env "MIN_ELO=2200" < jobs/parse_elo_ablation.sh  ← already running as parse_high_elo.sh
#
# Outputs: dataset_elo{MIN_ELO}_raw.pt
# Note: 2200+ is already parsed to dataset_high_elo_raw.pt — skip that submission.
#
# Next per condition:
#   bsub -env "MIN_ELO=2000" < jobs/reeval_elo_ablation.sh  (× 7 chunks)
#   bsub -env "MIN_ELO=2100" < jobs/reeval_elo_ablation.sh  (× 7 chunks)
#   bsub -env "MIN_ELO=2200" < jobs/reeval_elo_ablation.sh  (× 7 chunks, uses dataset_high_elo_raw.pt)

MIN_ELO="${MIN_ELO:-2000}"
MAX_GAMES=250000

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_2025-02.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_elo${MIN_ELO}_raw.pt"

echo "=== Parse ELO ablation: min_elo=${MIN_ELO}  max_games=${MAX_GAMES} ==="
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
    --max-games          ${MAX_GAMES} \
    --min-elo            ${MIN_ELO} \
    --positions-per-game 20 \
    --skip-opening       5 \
    --sampling           even \
    --no-strict          \
    --no-split           \
    --checkpoint-every   0

echo
echo "Done. Raw dataset: ${OUT_FILE}"
echo "Check draw rate: python3 src/check_draw_rate.py ${OUT_FILE}"
echo "Next: submit reeval_elo_ablation.sh (7 chunks, MIN_ELO=${MIN_ELO})"
