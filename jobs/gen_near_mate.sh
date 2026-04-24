#!/bin/bash
#BSUB -J gen_near_mate
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=4GB] select[hname!='n-62-21-89']"
#BSUB -W 12:00
#BSUB -o /zhome/81/b/206091/logs/gen_near_mate_%J.out
#BSUB -e /zhome/81/b/206091/logs/gen_near_mate_%J.err

# Generate near-mate positions (forced mate in ≤ 3 moves) from real game data.
# Pools FENs from low + mid + high band datasets (all position types), pre-filters
# to decisive positions (|v| ≥ 0.90), then confirms mate-in-3 with SF depth-18.
#
#   bsub < jobs/gen_near_mate.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

OUT="${BLACKHOLE}/dataset_near_mate.pt"

echo "=== Near-mate dataset generation ==="
echo "Sources: low + mid + high band SF-18 datasets"
echo "Filter:  mate-in ≤ 3, confirmed by SF depth-18"
echo "Out:     ${OUT}"
echo

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/gen_near_mate.py" \
    --sources \
        "${BLACKHOLE}/dataset_2021_06_low_sf18.pt" \
        "${BLACKHOLE}/dataset_2021_06_mid_sf18.pt" \
        "${BLACKHOLE}/dataset_2021_06_high_sf18.pt" \
    --out        "${OUT}" \
    --stockfish  "${HOME_DIR}/bin/stockfish" \
    --depth      18 \
    --n          220000 \
    --max-mate   3 \
    --workers    16 \
    --seed       42

echo
echo "Done. Dataset: ${OUT}"
