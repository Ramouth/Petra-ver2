#!/bin/bash
#BSUB -J gen_material_imbalance
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=2GB] select[hname!='n-62-21-89']"
#BSUB -W 12:00
#BSUB -o /zhome/81/b/206091/logs/gen_material_imbalance_%J.out
#BSUB -e /zhome/81/b/206091/logs/gen_material_imbalance_%J.err

# Generate material imbalance positions for the Phase 1.5 bootstrap.
# 10 combos (5 piece types x 2 sides), 20k target per combo = ~170k positions
# after filtering. Source: mid-band SF-18 dataset (realistic board positions).
#
# Runtime estimate: ~170k positions at 10 pos/s with 16 workers = ~5h
#
#   bsub < jobs/gen_material_imbalance.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

SOURCE="${BLACKHOLE}/dataset_2021_06_mid_sf18.pt"
OUT="${BLACKHOLE}/dataset_material_imbalance.pt"

echo "=== Material imbalance dataset generation ==="
echo "Source:  ${SOURCE}"
echo "Out:     ${OUT}"
echo "Combos:  5 piece types x 2 sides = 10"
echo "Target:  20,000 per combo → ~170k after filtering"
echo

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/gen_material_imbalance.py" \
    --source      "${SOURCE}" \
    --out         "${OUT}" \
    --stockfish   "${HOME_DIR}/bin/stockfish" \
    --depth       18 \
    --n-per-combo 20000 \
    --workers     16 \
    --seed        42

echo
echo "Done. Dataset: ${OUT}"
echo "Next: merge with endgame + mid datasets, then train."
