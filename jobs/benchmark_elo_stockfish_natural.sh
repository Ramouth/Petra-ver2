#!/bin/bash
set -euo pipefail
#BSUB -J benchmark_elo_natural
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 8:00
#BSUB -o /zhome/81/b/206091/logs/benchmark_elo_natural_%J.out
#BSUB -e /zhome/81/b/206091/logs/benchmark_elo_natural_%J.err

# Calibrate natural's playing strength against Stockfish skill levels.
# Provides an absolute ELO number for the paper's results table.
#
# 100 games per skill level (3 levels = 300 games, ~5-7 hours wall).
# Stockfish skill levels chosen to bracket expected strength:
#   skill=0  (~1320 ELO)  — should win most
#   skill=5  (~1500 ELO)  — should win majority
#   skill=10 (~1750 ELO)  — likely 50/50 territory
#
# If natural scores >70% at skill=10, run skill=15 next.
# If natural scores <30% at skill=5, run skill=0 only.
#
#   bsub < jobs/benchmark_elo_stockfish_natural.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODEL="${HOME_DIR}/Petra-ver2/models/natural/best.pt"
STOCKFISH="${HOME_DIR}/bin/stockfish"
OUT="${HOME_DIR}/Petra-ver2/results/elo_natural.json"

if [ ! -f "${MODEL}" ];     then echo "ERROR: not found: ${MODEL}";     exit 1; fi
if [ ! -x "${STOCKFISH}" ]; then echo "ERROR: not found: ${STOCKFISH}"; exit 1; fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

mkdir -p "$(dirname "${OUT}")"

python3 -u "${SRC}/benchmark_elo_stockfish.py" \
    --model         "${MODEL}" \
    --stockfish     "${STOCKFISH}" \
    --skill-levels  0 5 10 \
    --games         100 \
    --n-sim         100 \
    --think-time-ms 100 \
    --out           "${OUT}"

echo
echo "Done. Results: ${OUT}"
