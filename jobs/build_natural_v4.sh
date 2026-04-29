#!/bin/bash
set -euo pipefail
#BSUB -J build_natural_v4
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=64GB] span[hosts=1]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/build_natural_v4_%J.out
#BSUB -e /zhome/81/b/206091/logs/build_natural_v4_%J.err

# Build natural v4: AlphaZero-style supervision (raw outcome values, no SF
# eval as teacher). Three disjoint Lichess months pooled and engineered to
# 5M positions @ 7.5% drawn. FEN-deduped as defense in depth.
#
#   bsub < jobs/build_natural_v4.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

OUT="${BLACKHOLE}/dataset_natural_v4.pt"

echo "=== Building natural v4 (outcome supervision, 5M @ 7.5% drawn) ==="
echo "Out:   ${OUT}"
echo

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/build_natural_v4.py" \
    --sources \
        "${BLACKHOLE}/dataset_2021_06_all_sf18.pt" \
        "${BLACKHOLE}/dataset_2023_03_sf18.pt" \
        "${BLACKHOLE}/dataset_2025_01.pt" \
    --target-size      5000000 \
    --target-draw-frac 0.075 \
    --val-frac         0.05 \
    --out              "${OUT}"

echo
echo "Done."
