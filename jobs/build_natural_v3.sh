#!/bin/bash
set -euo pipefail
#BSUB -J build_natural_v3
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=64GB] span[hosts=1]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/build_natural_v3_%J.out
#BSUB -e /zhome/81/b/206091/logs/build_natural_v3_%J.err

# Build natural v3: 4.5M positions @ 7.5% drawn, pooled from all SF-reeval'd
# high/mid-ELO sources. Engineered draw fraction by oversampling drawn and
# undersampling decisive.
#
# Skipping low-quality sources (low_elo, low-band 2021_06) to keep play
# strength signal high.
#
#   bsub < jobs/build_natural_v3.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

OUT="${BLACKHOLE}/dataset_natural_v3.pt"

echo "=== Building natural v3 dataset (4.5M @ 7.5% drawn) ==="
echo "Out:   ${OUT}"
echo

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/build_natural_v3.py" \
    --sources \
        "${BLACKHOLE}/dataset_elo2100_sf18.pt" \
        "${BLACKHOLE}/dataset_elo2200_sf18.pt" \
        "${BLACKHOLE}/dataset_elo2000_sf18.pt" \
        "${BLACKHOLE}/dataset_2021_06_all_sf18.pt" \
        "${BLACKHOLE}/dataset_2021_06_high_sf18.pt" \
        "${BLACKHOLE}/dataset_2023_03_sf18.pt" \
    --target-size      4500000 \
    --target-draw-frac 0.075 \
    --val-frac         0.05 \
    --out              "${OUT}"

echo
echo "Done."
