#!/bin/bash
set -euo pipefail
#BSUB -J build_poc_battery
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/build_poc_battery_%J.out
#BSUB -e /zhome/81/b/206091/logs/build_poc_battery_%J.err

# Build the PoC drawness gate corpus from 2021_06 high-elo Lichess data.
# Filters to dead-even middlegame positions (|SF|<0.2, ply 20-80), with a
# ±2-ply blunder filter, balanced 2500 drawn + 2500 decisive (one per game).
#
# 2021_06 is leakage-safe: natural and natural_v2 were trained on 2025-derived
# elo2000/2100/2200 datasets — different month, different elo bands.
#
#   bsub < jobs/build_poc_battery.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

INPUT="${BLACKHOLE}/dataset_2021_06_high_sf18.pt"
OUT="${HOME_DIR}/Petra-ver2/data/poc_battery_v0.pt"

echo "=== Build PoC drawness gate corpus ==="
echo "Source : ${INPUT}"
echo "Out    : ${OUT}"
echo

if [ ! -f "${INPUT}" ]; then
    echo "ERROR: input not found: ${INPUT}"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/build_poc_battery.py" \
    --input         "${INPUT}" \
    --out           "${OUT}" \
    --ply-min       20 \
    --ply-max       80 \
    --eval-thresh   0.2 \
    --blunder-window 2 \
    --blunder-delta  0.4 \
    --per-game-cap   1 \
    --n-per-class    2500 \
    --seed           42

echo
echo "Done. Battery: ${OUT}"
echo "Next: write src/poc_battery_eval.py and run on natural + natural_v2"
