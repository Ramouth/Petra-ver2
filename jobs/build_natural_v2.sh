#!/bin/bash
set -euo pipefail
#BSUB -J build_natural_v2
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/build_natural_v2_%J.out
#BSUB -e /zhome/81/b/206091/logs/build_natural_v2_%J.err

# Build natural v2 dataset:
#   base = full elo2100_sf18 (all 2100+ ELO games, all outcomes)
#   draw = drawn-only subset of elo2200_sf18 (focused drawness signal)
#
#   bsub < jobs/build_natural_v2.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

BASE="${BLACKHOLE}/dataset_elo2100_sf18.pt"
DRAW="${BLACKHOLE}/dataset_elo2200_sf18.pt"
OUT="${BLACKHOLE}/dataset_natural_v2.pt"

echo "=== Building natural v2 dataset ==="
echo "Base:  ${BASE}"
echo "Draw:  ${DRAW}"
echo "Out:   ${OUT}"
echo

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/build_natural_v2.py" \
    --base-dataset "${BASE}" \
    --draw-dataset "${DRAW}" \
    --out          "${OUT}"

echo
echo "Done."
