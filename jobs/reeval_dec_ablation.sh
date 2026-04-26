#!/bin/bash
#BSUB -J reeval_dec_ablation_c${CHUNK_IDX:-0}
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB] select[hname!='n-62-21-89']"
#BSUB -W 12:00
#BSUB -o /zhome/81/b/206091/logs/reeval_dec_ablation_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_dec_ablation_%J.err

# SF depth-18 reeval for decisiveness ablation. Submit all 7 chunks in parallel:
#
#   bsub -env "CHUNK_IDX=0" < jobs/reeval_dec_ablation.sh
#   bsub -env "CHUNK_IDX=1" < jobs/reeval_dec_ablation.sh
#   bsub -env "CHUNK_IDX=2" < jobs/reeval_dec_ablation.sh
#   bsub -env "CHUNK_IDX=3" < jobs/reeval_dec_ablation.sh
#   bsub -env "CHUNK_IDX=4" < jobs/reeval_dec_ablation.sh
#   bsub -env "CHUNK_IDX=5" < jobs/reeval_dec_ablation.sh
#   bsub -env "CHUNK_IDX=6" < jobs/reeval_dec_ablation.sh
#
# When all 7 complete: submit reeval_dec_ablation_merge.sh for each DEC_LEVEL.

CHUNK_IDX="${CHUNK_IDX:-0}"
N_CHUNKS=7
N=5000000

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/dataset_dec_ablation_raw.pt"
PARTIAL_OUT="${BLACKHOLE}/reeval_dec_ablation_d18_part${CHUNK_IDX}.pt"

echo "=== SF depth-18 reeval: dec_ablation chunk ${CHUNK_IDX}/${N_CHUNKS} ==="
echo "In:      ${IN_FILE}"
echo "Partial: ${PARTIAL_OUT}"
echo

if [ ! -f "${IN_FILE}" ]; then
    echo "ERROR: input not found: ${IN_FILE}"
    echo "Run parse_dec_ablation.sh first."
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/reeval_stockfish.py" \
    --dataset     "${IN_FILE}" \
    --stockfish   "${HOME_DIR}/bin/stockfish" \
    --depth       18 \
    --n           ${N} \
    --seed        42 \
    --workers     16 \
    --chunk-idx   ${CHUNK_IDX} \
    --n-chunks    ${N_CHUNKS} \
    --partial-out "${PARTIAL_OUT}"

echo
echo "Done. Partial: ${PARTIAL_OUT}"
