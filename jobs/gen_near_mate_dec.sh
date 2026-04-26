#!/bin/bash
#BSUB -J gen_near_mate_dec
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=4GB] select[hname!='n-62-21-89']"
#BSUB -W 12:00
#BSUB -o /zhome/81/b/206091/logs/gen_near_mate_dec_%J.out
#BSUB -e /zhome/81/b/206091/logs/gen_near_mate_dec_%J.err

# Generate near-mate positions (forced mate in ≤ 3 moves) from dec_ablation data.
# Source: dataset_dec00_sf18.pt (all positions, no decisiveness filter).
# Submit after reeval_dec_ablation_merge.sh DEC_LEVEL=0.0 completes.
#
#   bsub < jobs/gen_near_mate_dec.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

SOURCE="${BLACKHOLE}/dataset_dec00_sf18.pt"
OUT="${BLACKHOLE}/dataset_near_mate_dec.pt"

echo "=== Near-mate dataset generation (dec_ablation source) ==="
echo "Source: ${SOURCE}"
echo "Filter: mate-in ≤ 3, confirmed by SF depth-18"
echo "Out:    ${OUT}"
echo

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/gen_near_mate.py" \
    --sources    "${SOURCE}" \
    --out        "${OUT}" \
    --stockfish  "${HOME_DIR}/bin/stockfish" \
    --depth      18 \
    --n          220000 \
    --max-mate   3 \
    --workers    16 \
    --seed       42

echo
echo "Done. Dataset: ${OUT}"
