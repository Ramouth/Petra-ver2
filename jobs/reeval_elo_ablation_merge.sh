#!/bin/bash
set -euo pipefail
#BSUB -J reeval_elo${MIN_ELO:-2000}_merge
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/reeval_elo_ablation_merge_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_elo_ablation_merge_%J.err

# Merge 7 reeval chunks into the final ELO ablation dataset.
# Submit per band after all 7 chunks for that band complete.
#
#   bsub -env "MIN_ELO=2000" < jobs/reeval_elo_ablation_merge.sh
#   bsub -env "MIN_ELO=2100" < jobs/reeval_elo_ablation_merge.sh
#   bsub -env "MIN_ELO=2200" < jobs/reeval_elo_ablation_merge.sh
#
# Outputs: dataset_elo{MIN_ELO}_sf18.pt
# N=5,000,000 must match what the chunk jobs used (recorded in partials).

MIN_ELO="${MIN_ELO:-2000}"
N=5000000

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

if [ "${MIN_ELO}" = "2200" ]; then
    IN_FILE="${BLACKHOLE}/dataset_high_elo_raw.pt"
else
    IN_FILE="${BLACKHOLE}/dataset_elo${MIN_ELO}_raw.pt"
fi
OUT_FILE="${BLACKHOLE}/dataset_elo${MIN_ELO}_sf18.pt"

echo "=== ELO ablation merge: MIN_ELO=${MIN_ELO}  n=${N} ==="
echo "In:  ${IN_FILE}"
echo "Out: ${OUT_FILE}"
echo

for i in 0 1 2 3; do
    PART="${BLACKHOLE}/reeval_elo${MIN_ELO}_d18_part${i}.pt"
    if [ ! -f "${PART}" ]; then
        echo "ERROR: missing partial: ${PART}"
        exit 1
    fi
done

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/reeval_stockfish.py" \
    --dataset                      "${IN_FILE}" \
    --out                          "${OUT_FILE}" \
    --merge \
        "${BLACKHOLE}/reeval_elo${MIN_ELO}_d18_part0.pt" \
        "${BLACKHOLE}/reeval_elo${MIN_ELO}_d18_part1.pt" \
        "${BLACKHOLE}/reeval_elo${MIN_ELO}_d18_part2.pt" \
        "${BLACKHOLE}/reeval_elo${MIN_ELO}_d18_part3.pt" \
    --n                            ${N} \
    --seed                         42 \
    --min-decisive                 0.0 \
    --max-pieces                   32 \
    --derive-drawness-from-outcome \
    --drawness-sf-threshold        0.18 \
    --drawness-target              0.8 \
    --allow-partial-coverage

echo
echo "Done. Dataset: ${OUT_FILE}"
echo "Next: bsub -env \"MIN_ELO=${MIN_ELO}\" < jobs/train_elo_ablation_gpu.sh"
