#!/bin/bash
#BSUB -J finish_dataset
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/finish_dataset_%J.out
#BSUB -e /zhome/81/b/206091/logs/finish_dataset_%J.err

MONTH="${MONTH:-03}"
YEAR="${YEAR:-2023}"
BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

CKPT_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}.ckpt.pt"
OUT_FILE="${BLACKHOLE}/dataset_${YEAR}_${MONTH}.pt"

echo "=== Finishing dataset from checkpoint ==="
echo "Checkpoint: ${CKPT_FILE}"
echo "Output:     ${OUT_FILE}"
echo

if [ ! -f "${CKPT_FILE}" ]; then
    echo "ERROR: checkpoint not found at ${CKPT_FILE}"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/data.py" \
    --from-checkpoint "${CKPT_FILE}" \
    --out             "${OUT_FILE}" \
    --positions-per-game 20 \
    --skip-opening       5 \
    --sampling           even \
    --no-strict

echo
echo "Done. Output: ${OUT_FILE}"
echo "Next step: submit 3 chunk jobs:"
echo "  MONTH=${MONTH} YEAR=${YEAR} CHUNK_IDX=0 bsub < jobs/reeval_depth20.sh"
echo "  MONTH=${MONTH} YEAR=${YEAR} CHUNK_IDX=1 bsub < jobs/reeval_depth20.sh"
echo "  MONTH=${MONTH} YEAR=${YEAR} CHUNK_IDX=2 bsub < jobs/reeval_depth20.sh"
