#!/bin/bash
set -euo pipefail
#BSUB -J benchmark_elo_dr03
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 8:00
#BSUB -o /zhome/81/b/206091/logs/benchmark_elo_dr03_%J.out
#BSUB -e /zhome/81/b/206091/logs/benchmark_elo_dr03_%J.err

# Calibrate dr03 (drawness fork champion) playing strength against
# Stockfish skill levels. Gives an absolute ELO for the petra-drawness
# track to compare with natural's number.
#
# Same skill-level bracket as natural's benchmark for direct comparison.
# Expected: dr03 lands ~50-100 ELO below natural.
#
#   bsub < jobs/benchmark_elo_stockfish_dr03.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
MODEL="${HOME_DIR}/Petra-ver2/models/drawness_full_natural_init_dr03/best.pt"
STOCKFISH="${HOME_DIR}/bin/stockfish"
OUT="${HOME_DIR}/Petra-ver2/results/elo_dr03.json"

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
