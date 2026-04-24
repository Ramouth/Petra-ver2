#!/bin/bash
#BSUB -J reeval_phase15_mid_merge
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/reeval_phase15_mid_merge_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_phase15_mid_merge_%J.err

# Merge 3 SF reeval chunks into the final Phase 1.5 mid dataset.
# Submit after all 3 reeval_phase15_mid chunks complete.
#
#   bsub < jobs/reeval_phase15_mid_merge.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/dataset_phase15_mid.pt"
OUT_FILE="${BLACKHOLE}/dataset_phase15_mid_sf18.pt"

for i in 0 1 2; do
    PART="${BLACKHOLE}/reeval_phase15_mid_d18_part${i}.pt"
    if [ ! -f "${PART}" ]; then
        echo "ERROR: missing partial: ${PART}"
        exit 1
    fi
done

echo "=== Merging phase15 mid reeval chunks → ${OUT_FILE} ==="
echo

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/reeval_stockfish.py" \
    --dataset                      "${IN_FILE}" \
    --out                          "${OUT_FILE}" \
    --merge \
        "${BLACKHOLE}/reeval_phase15_mid_d18_part0.pt" \
        "${BLACKHOLE}/reeval_phase15_mid_d18_part1.pt" \
        "${BLACKHOLE}/reeval_phase15_mid_d18_part2.pt" \
    --n                            2100000 \
    --seed                         42 \
    --min-decisive                 0.0 \
    --max-pieces                   32 \
    --derive-drawness-from-outcome \
    --drawness-game-level \
    --drawness-sf-threshold        0.11 \
    --drawness-target              0.8 \
    --allow-partial-coverage

echo
echo "Done. Final dataset: ${OUT_FILE}"
echo "Next: gen_endgame_drawness.sh and check dataset_material_imbalance.pt, then train Phase 1.5."
