#!/bin/bash
#BSUB -J reeval_dec_probe_${DEC_FRAC:-0.00}
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/reeval_dec_probe_merge_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_dec_probe_merge_%J.err

# Probe experiment: vary fraction of decisive games (game result ≠ draw),
# holding N=1M and SF labels constant. Reuses the dec_ablation reeval partials.
#
# DEC_FRAC: fraction of final positions from decisive games (0.00 to 1.00)
#
#   bsub -env "DEC_FRAC=0.00" < jobs/reeval_dec_probe_merge.sh
#   bsub -env "DEC_FRAC=0.25" < jobs/reeval_dec_probe_merge.sh
#   bsub -env "DEC_FRAC=0.50" < jobs/reeval_dec_probe_merge.sh
#   bsub -env "DEC_FRAC=0.75" < jobs/reeval_dec_probe_merge.sh
#   bsub -env "DEC_FRAC=1.00" < jobs/reeval_dec_probe_merge.sh
#
# When all 5 complete: submit train_dec_probe.sh for each DEC_FRAC.

DEC_FRAC="${DEC_FRAC:-0.50}"
N=5000000
N_OUTPUT=1000000

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/dataset_dec_ablation_raw.pt"
TAG="${DEC_FRAC//./}"           # "0.25" → "025", "1.00" → "100"
OUT_FILE="${BLACKHOLE}/dataset_dec_probe_frac${TAG}_sf18.pt"

echo "=== Dec probe merge: DEC_FRAC=${DEC_FRAC}  n_output=${N_OUTPUT} ==="
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
    --min-decisive                 0.0 \
    --max-pieces                   32 \
    --decisive-game-fraction       ${DEC_FRAC} \
    --n-output                     ${N_OUTPUT} \
    --derive-drawness-from-outcome \
    --drawness-game-level \
    --drawness-sf-threshold        0.11 \
    --drawness-target              0.8 \
    --allow-partial-coverage

echo
echo "Done. Dataset: ${OUT_FILE}"
echo "Next: bsub -env \"DEC_FRAC=${DEC_FRAC}\" < jobs/train_dec_probe.sh"
