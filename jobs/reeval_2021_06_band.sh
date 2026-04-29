#!/bin/bash
#BSUB -J reeval_2021_06_${BAND:-mid}
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB] select[hname!='n-62-21-89']"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/reeval_2021_06_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_2021_06_%J.err

# SF depth-18 reeval for 2021-06 banded datasets (~662k positions per band).
# Each band uses 3 chunks — submit all 3 in parallel, then run the merge job.
# Use -env to pass vars to LSF:
#
#   bsub -env "BAND=mid,CHUNK_IDX=0" < jobs/reeval_2021_06_band.sh
#   bsub -env "BAND=mid,CHUNK_IDX=1" < jobs/reeval_2021_06_band.sh
#   bsub -env "BAND=mid,CHUNK_IDX=2" < jobs/reeval_2021_06_band.sh
#   bsub -env "BAND=mid"             < jobs/reeval_2021_06_merge.sh
#
# Repeat with BAND=low and BAND=high.

BAND="${BAND:-mid}"
CHUNK_IDX="${CHUNK_IDX:-0}"
N_CHUNKS=3
N=700000

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/dataset_2021_06_${BAND}.pt"
PARTIAL_OUT="${BLACKHOLE}/reeval_2021_06_${BAND}_d18_part${CHUNK_IDX}.pt"

echo "=== SF depth-18 reeval: 2021-06 band=${BAND} chunk ${CHUNK_IDX}/${N_CHUNKS} ==="
echo "In:      ${IN_FILE}"
echo "Partial: ${PARTIAL_OUT}"
echo

if [ ! -f "${IN_FILE}" ]; then
    echo "ERROR: input dataset not found: ${IN_FILE}"
    echo "Run BAND=${BAND} bsub < jobs/parse_2021_06_band.sh first"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/reeval_stockfish.py" \
    --dataset    "${IN_FILE}" \
    --stockfish  "${HOME_DIR}/bin/stockfish" \
    --depth      18 \
    --n          ${N} \
    --seed       42 \
    --workers    16 \
    --chunk-idx  ${CHUNK_IDX} \
    --n-chunks   ${N_CHUNKS} \
    --partial-out "${PARTIAL_OUT}"

echo
echo "Done. Partial saved: ${PARTIAL_OUT}"
echo "When all 3 chunks done: BAND=${BAND} bsub < jobs/reeval_2021_06_merge.sh"
