#!/bin/bash
#BSUB -J reeval_2021_06_merge
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/reeval_2021_06_merge_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_2021_06_merge_%J.err

# Merge 3 reeval chunks for a 2021-06 band into a final SF-labelled dataset.
# Submit after all 3 chunks for the band complete:
#
#   bsub -env "BAND=mid"  < jobs/reeval_2021_06_merge.sh
#   bsub -env "BAND=low"  < jobs/reeval_2021_06_merge.sh
#   bsub -env "BAND=high" < jobs/reeval_2021_06_merge.sh

BAND="${BAND:-mid}"
N_CHUNKS=3

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/dataset_2021_06_${BAND}.pt"
OUT_FILE="${BLACKHOLE}/dataset_2021_06_${BAND}_sf18.pt"

PARTIALS=""
for i in 0 1 2; do
    PART="${BLACKHOLE}/reeval_2021_06_${BAND}_d18_part${i}.pt"
    if [ ! -f "${PART}" ]; then
        echo "ERROR: missing partial: ${PART}"
        exit 1
    fi
    PARTIALS="${PARTIALS} ${PART}"
done

echo "=== Merging reeval chunks for 2021-06 band=${BAND} → ${OUT_FILE} ==="
echo "Partials:${PARTIALS}"
echo

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/reeval_stockfish.py" \
    --dataset                      "${IN_FILE}" \
    --out                          "${OUT_FILE}" \
    --merge                        ${PARTIALS} \
    --n                            700000 \
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
