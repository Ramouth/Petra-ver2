#!/bin/bash
#BSUB -J gen_near_mate
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=2GB] select[hname!='n-62-21-89']"
#BSUB -W 12:00
#BSUB -o /zhome/81/b/206091/logs/gen_near_mate_%J.out
#BSUB -e /zhome/81/b/206091/logs/gen_near_mate_%J.err

# Generate 220k near-mate positions (forced mate in ≤ 7 moves).
# Sources KQ vs K, KR vs K, KQ vs KR endgame positions and filters
# with Stockfish depth-18 for confirmed forced-mate positions.
#
#   bsub < jobs/gen_near_mate.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

OUT="${BLACKHOLE}/dataset_near_mate.pt"

echo "=== Near-mate dataset generation ==="
echo "Target:  220,000 positions (mate-in ≤ 7)"
echo "Out:     ${OUT}"
echo

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/gen_near_mate.py" \
    --out        "${OUT}" \
    --stockfish  "${HOME_DIR}/bin/stockfish" \
    --depth      18 \
    --n          220000 \
    --max-mate   7 \
    --workers    16 \
    --seed       42

echo
echo "Done. Dataset: ${OUT}"
