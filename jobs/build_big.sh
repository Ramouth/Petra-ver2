#!/bin/bash
set -euo pipefail
#BSUB -J build_big
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=64GB] span[hosts=1]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/build_big_%J.out
#BSUB -e /zhome/81/b/206091/logs/build_big_%J.err

# Build dataset_big.pt — outcome-supervised, 5M positions @ 7.5% drawn,
# from the freshly parsed 2025-04 month at min_elo=2400.
#
# Reuses build_natural_v4.py (single source, FEN-deduped). v4 was never
# run; we now use its same machinery on a clean fresh-month source with
# zero overlap with natural / natural_v2 / 2021_06_high test corpus.
#
#   bsub < jobs/build_big.sh   # after parse_2025_04_2400 completes

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

SOURCE="${BLACKHOLE}/dataset_2025_04.pt"
OUT="${BLACKHOLE}/dataset_big.pt"

echo "=== Building dataset_big (outcome supervision, 5M @ 7.5% drawn) ==="
echo "Source: ${SOURCE}"
echo "Out:    ${OUT}"
echo

if [ ! -f "${SOURCE}" ]; then
    echo "ERROR: source not found: ${SOURCE}"
    echo "Run jobs/parse_2025_04_2400.sh first."
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/build_natural_v4.py" \
    --sources           "${SOURCE}" \
    --target-size       5000000 \
    --target-draw-frac  0.075 \
    --val-frac          0.05 \
    --out               "${OUT}"

echo
echo "Done. Dataset: ${OUT}"
echo "Next: bsub < jobs/train_big.sh"
