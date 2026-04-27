#!/bin/bash
#BSUB -J train_dec_probe_${DEC_FRAC:-0.50}
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_dec_probe_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_dec_probe_%J.err

# Train dec probe condition. All 5 can run in parallel once
# their respective merge datasets are ready.
#
#   bsub -env "DEC_FRAC=0.00" < jobs/train_dec_probe.sh
#   bsub -env "DEC_FRAC=0.25" < jobs/train_dec_probe.sh
#   bsub -env "DEC_FRAC=0.50" < jobs/train_dec_probe.sh
#   bsub -env "DEC_FRAC=0.75" < jobs/train_dec_probe.sh
#   bsub -env "DEC_FRAC=1.00" < jobs/train_dec_probe.sh

DEC_FRAC="${DEC_FRAC:-0.50}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

TAG="${DEC_FRAC//./}"
DATASET="${BLACKHOLE}/dataset_dec_probe_frac${TAG}_sf18.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/phase15_mid_no_endgame/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/dec_probe_frac${TAG}"

echo "=== Dec probe train: DEC_FRAC=${DEC_FRAC} ==="
echo "Dataset:    ${DATASET}"
echo "Init model: ${INIT_MODEL}"
echo "Out:        ${OUT_DIR}"
echo

for F in "${DATASET}" "${INIT_MODEL}"; do
    if [ ! -f "${F}" ]; then echo "ERROR: not found: ${F}"; exit 1; fi
done

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

python3 -u "${SRC}/train.py" \
    --dataset       "${DATASET}" \
    --init-model    "${INIT_MODEL}" \
    --out           "${OUT_DIR}" \
    --policy-weight 0.0 \
    --rank-reg      0.5 \
    --draw-reg      0.0 \
    --weight-decay  5e-4 \
    --lr            3e-4 \
    --epochs        30 \
    --geo-patience  5 \
    --num-workers   0

echo
echo "Done. Checkpoint: ${OUT_DIR}/best.pt"
echo "Next: bsub -env \"DEC_FRAC=${DEC_FRAC}\" < jobs/eval_dec_probe.sh"
