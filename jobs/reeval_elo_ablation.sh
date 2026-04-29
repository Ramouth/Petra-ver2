#!/bin/bash
#BSUB -J reeval_elo${MIN_ELO:-2000}_c${CHUNK_IDX:-0}
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB] select[hname!='n-62-21-89']"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/reeval_elo_ablation_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_elo_ablation_%J.err

# SF depth-18 reeval for ELO ablation. Submit all 7 chunks per band in parallel:
#
#   bsub -env "MIN_ELO=2000,CHUNK_IDX=0" < jobs/reeval_elo_ablation.sh
#   bsub -env "MIN_ELO=2000,CHUNK_IDX=1" < jobs/reeval_elo_ablation.sh
#   ... (× 7 for each MIN_ELO in 2000, 2100, 2200)
#
# 2200+ input: dataset_high_elo_raw.pt (not dataset_elo2200_raw.pt).
# Partials: reeval_elo{MIN_ELO}_d18_part{CHUNK_IDX}.pt
#
# When all 7 chunks per band complete: submit reeval_elo_ablation_merge.sh.

MIN_ELO="${MIN_ELO:-2000}"
CHUNK_IDX="${CHUNK_IDX:-0}"
N_CHUNKS=4
N=5000000

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

if [ "${MIN_ELO}" = "2200" ]; then
    IN_FILE="${BLACKHOLE}/dataset_high_elo_raw.pt"
else
    IN_FILE="${BLACKHOLE}/dataset_elo${MIN_ELO}_raw.pt"
fi
PARTIAL_OUT="${BLACKHOLE}/reeval_elo${MIN_ELO}_d18_part${CHUNK_IDX}.pt"

echo "=== SF depth-18 reeval: ELO ${MIN_ELO}+ chunk ${CHUNK_IDX}/${N_CHUNKS} ==="
echo "In:      ${IN_FILE}"
echo "Partial: ${PARTIAL_OUT}"
echo

if [ ! -f "${IN_FILE}" ]; then
    echo "ERROR: input not found: ${IN_FILE}"
    echo "Run parse_elo_ablation.sh (or parse_high_elo.sh for 2200+) first."
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
