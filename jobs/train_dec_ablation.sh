#!/bin/bash
#BSUB -J train_dec_${DEC_LEVEL:-0.0}
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_dec_ablation_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_dec_ablation_%J.err

# Train decisiveness ablation condition. All 4 can run in parallel once
# their respective merge datasets are ready.
#
#   bsub -env "DEC_LEVEL=0.0" < jobs/train_dec_ablation.sh
#   bsub -env "DEC_LEVEL=0.3" < jobs/train_dec_ablation.sh
#   bsub -env "DEC_LEVEL=0.5" < jobs/train_dec_ablation.sh
#   bsub -env "DEC_LEVEL=0.7" < jobs/train_dec_ablation.sh

DEC_LEVEL="${DEC_LEVEL:-0.0}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

TAG="${DEC_LEVEL//./}"
DATASET="${BLACKHOLE}/dataset_dec${TAG}_sf18.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/phase15_mid_no_endgame/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/dec${TAG}"

echo "=== Decisiveness ablation train: DEC_LEVEL=${DEC_LEVEL} ==="
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
echo "Next: bsub -env \"DEC_LEVEL=${DEC_LEVEL}\" < jobs/eval_dec_ablation.sh"
