#!/bin/bash
#BSUB -J parse_dec_merge
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 6:00
#BSUB -o /zhome/81/b/206091/logs/parse_dec_ablation_merge_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_dec_ablation_merge_%J.err

# Merge 4 parse chunks into one validated, train/val-split dataset.
# Submit after all 4 parse_dec_ablation.sh chunks complete.
#
#   bsub < jobs/parse_dec_ablation_merge.sh
#
# Next: submit 6 reeval chunks in parallel:
#   for i in 0 1 2 3 4 5; do bsub -env "CHUNK_IDX=$i" < jobs/reeval_dec_ablation.sh; done

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

OUT_FILE="${BLACKHOLE}/dataset_dec_ablation_raw.pt"

echo "=== Merging parse chunks 0-4 → ${OUT_FILE} ==="
echo

for i in 0 1 2 3 4; do
    PART="${BLACKHOLE}/dataset_dec_ablation_raw_c${i}.pt"
    if [ ! -f "${PART}" ]; then
        echo "ERROR: missing chunk: ${PART}"
        exit 1
    fi
done

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/data.py" \
    --merge-raw \
        "${BLACKHOLE}/dataset_dec_ablation_raw_c0.pt" \
        "${BLACKHOLE}/dataset_dec_ablation_raw_c1.pt" \
        "${BLACKHOLE}/dataset_dec_ablation_raw_c2.pt" \
        "${BLACKHOLE}/dataset_dec_ablation_raw_c3.pt" \
        "${BLACKHOLE}/dataset_dec_ablation_raw_c4.pt" \
    --out       "${OUT_FILE}" \
    --no-strict

echo
echo "Done. Merged dataset: ${OUT_FILE}"
echo "Next: submit 7 reeval chunks:"
echo "  for i in 0 1 2 3 4 5 6; do bsub -env \"CHUNK_IDX=\$i\" < jobs/reeval_dec_ablation.sh; done"
