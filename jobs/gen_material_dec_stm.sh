#!/bin/bash
#BSUB -J gen_material_dec_stm
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=2GB] select[hname!='n-62-21-89']"
#BSUB -W 12:00
#BSUB -o /zhome/81/b/206091/logs/gen_material_dec_stm_%J.out
#BSUB -e /zhome/81/b/206091/logs/gen_material_dec_stm_%J.err

# Generate material imbalance positions where STM is down one piece.
# 5 piece types × STM side only, 20k target per combo = ~85k positions.
# Source: dataset_dec00_sf18.pt (all positions, no decisiveness filter).
# Submit after reeval_dec_ablation_merge.sh DEC_LEVEL=0.0 completes.
# Run in parallel with gen_material_dec_opp.sh.
#
#   bsub < jobs/gen_material_dec_stm.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

SOURCE="${BLACKHOLE}/dataset_dec00_sf18.pt"
OUT="${BLACKHOLE}/dataset_material_dec_stm.pt"

echo "=== Material imbalance dataset (STM down one piece, dec_ablation source) ==="
echo "Source:  ${SOURCE}"
echo "Out:     ${OUT}"
echo "Combos:  5 piece types × STM = 5"
echo "Target:  20,000 per combo → ~85k after filtering"
echo

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/gen_material_imbalance.py" \
    --source      "${SOURCE}" \
    --out         "${OUT}" \
    --stockfish   "${HOME_DIR}/bin/stockfish" \
    --depth       18 \
    --n-per-combo 20000 \
    --sides       stm \
    --workers     16 \
    --seed        42

echo
echo "Done. Dataset: ${OUT}"
