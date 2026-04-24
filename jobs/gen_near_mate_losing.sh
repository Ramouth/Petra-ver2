#!/bin/bash
#BSUB -J gen_near_mate_losing
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=4GB] select[hname!='n-62-21-89']"
#BSUB -W 12:00
#BSUB -o /zhome/81/b/206091/logs/gen_near_mate_losing_%J.out
#BSUB -e /zhome/81/b/206091/logs/gen_near_mate_losing_%J.err

# Generate 'being mated' positions to balance the near-mate dataset.
#
# Two paths per candidate FEN:
#   Derived:  winning near-mate (mate ≤ 3) → push first move → losing position
#   Natural:  naturally-occurring losing positions (|mate| ≤ 5)
#
# Expected yield from the 310k pool: ~65-77k.
# If the WARNING fires, resubmit with --max-mate-win 5 (~100-125k yield).
#
#   bsub < jobs/gen_near_mate_losing.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

OUT="${BLACKHOLE}/dataset_near_mate_losing.pt"

echo "=== Near-mate losing dataset generation ==="
echo "Sources: low + mid + high band SF-18 datasets"
echo "Filter:  derived from winning mate ≤ 3  +  natural losing |mate| ≤ 5"
echo "Out:     ${OUT}"
echo

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/gen_near_mate_losing.py" \
    --sources \
        "${BLACKHOLE}/dataset_2021_06_low_sf18.pt" \
        "${BLACKHOLE}/dataset_2021_06_mid_sf18.pt" \
        "${BLACKHOLE}/dataset_2021_06_high_sf18.pt" \
    --out           "${OUT}" \
    --stockfish     "${HOME_DIR}/bin/stockfish" \
    --depth         18 \
    --n             100000 \
    --max-mate-win  3 \
    --max-mate-lose 5 \
    --workers       16 \
    --seed          42

echo
echo "Done. Dataset: ${OUT}"
echo "Next: merge dataset_near_mate.pt + dataset_near_mate_losing.pt before training."
