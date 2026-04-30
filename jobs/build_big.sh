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
# from two freshly parsed Lichess months at min_elo=2400 (2025-02 and
# 2025-04). Both are completely fresh sources, zero overlap with
# natural / natural_v2 training data or the 2021_06_high test corpus.
#
# Reuses build_natural_v4.py — outcome-as-value substitution + FEN dedup
# across both months. v4 was never run; this is the first use of its
# machinery on truly clean data.
#
#   bsub < jobs/build_big.sh   # after BOTH parse_2025_{02,04}_2400 complete

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

SOURCE_02="${BLACKHOLE}/dataset_2025_02.pt"
SOURCE_04="${BLACKHOLE}/dataset_2025_04.pt"
OUT="${BLACKHOLE}/dataset_big.pt"

echo "=== Building dataset_big (outcome supervision, 5M @ 7.5% drawn) ==="
echo "Sources:"
echo "  ${SOURCE_02}"
echo "  ${SOURCE_04}"
echo "Out: ${OUT}"
echo

for F in "${SOURCE_02}" "${SOURCE_04}"; do
    if [ ! -f "${F}" ]; then
        echo "ERROR: source not found: ${F}"
        echo "Run both parse_2025_02_2400.sh and parse_2025_04_2400.sh first."
        exit 1
    fi
done

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/build_natural_v4.py" \
    --sources           "${SOURCE_02}" "${SOURCE_04}" \
    --target-size       5000000 \
    --target-draw-frac  0.075 \
    --val-frac          0.05 \
    --out               "${OUT}"

echo
echo "Done. Dataset: ${OUT}"
echo "Next: bsub < jobs/train_big.sh"
