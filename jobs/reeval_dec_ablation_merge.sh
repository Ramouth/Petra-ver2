#!/bin/bash
#BSUB -J reeval_dec_merge_${DEC_LEVEL:-0.0}
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/reeval_dec_ablation_merge_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_dec_ablation_merge_%J.err

# Merge 6 reeval chunks into one dataset at a fixed decisiveness threshold.
# All 4 merge jobs can run in parallel once all 6 reeval chunks finish.
#
# DEC_LEVEL: 0.0 | 0.3 | 0.5 | 0.7
# n is fixed at 1,000,000 across all conditions (controlled experiment).
#
# NOTE: If fewer than 1,000,000 positions pass the filter (most likely for
# DEC_LEVEL=0.7), this job will fail loudly. Inspect the reeval output for
# the actual passing count and lower N uniformly across all conditions if needed.
#
#   bsub -env "DEC_LEVEL=0.0" < jobs/reeval_dec_ablation_merge.sh
#   bsub -env "DEC_LEVEL=0.3" < jobs/reeval_dec_ablation_merge.sh
#   bsub -env "DEC_LEVEL=0.5" < jobs/reeval_dec_ablation_merge.sh
#   bsub -env "DEC_LEVEL=0.7" < jobs/reeval_dec_ablation_merge.sh

DEC_LEVEL="${DEC_LEVEL:-0.0}"
N=1000000

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/dataset_dec_ablation_raw.pt"
TAG="${DEC_LEVEL//./}"           # "0.3" → "03", "0.7" → "07"
OUT_FILE="${BLACKHOLE}/dataset_dec${TAG}_sf18.pt"

echo "=== Decisiveness ablation merge: DEC_LEVEL=${DEC_LEVEL}  n=${N} ==="
echo "In:  ${IN_FILE}"
echo "Out: ${OUT_FILE}"
echo

for i in 0 1 2 3 4 5 6; do
    PART="${BLACKHOLE}/reeval_dec_ablation_d18_part${i}.pt"
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
        "${BLACKHOLE}/reeval_dec_ablation_d18_part0.pt" \
        "${BLACKHOLE}/reeval_dec_ablation_d18_part1.pt" \
        "${BLACKHOLE}/reeval_dec_ablation_d18_part2.pt" \
        "${BLACKHOLE}/reeval_dec_ablation_d18_part3.pt" \
        "${BLACKHOLE}/reeval_dec_ablation_d18_part4.pt" \
        "${BLACKHOLE}/reeval_dec_ablation_d18_part5.pt" \
        "${BLACKHOLE}/reeval_dec_ablation_d18_part6.pt" \
    --n                            ${N} \
    --seed                         42 \
    --min-decisive                 ${DEC_LEVEL} \
    --max-pieces                   32 \
    --derive-drawness-from-outcome \
    --drawness-game-level \
    --drawness-sf-threshold        0.11 \
    --drawness-target              0.8 \
    --allow-partial-coverage

echo
echo "Done. Dataset: ${OUT_FILE}"
echo "Next: bsub -env \"DEC_LEVEL=${DEC_LEVEL}\" < jobs/train_dec_ablation.sh"
