#!/bin/bash
#BSUB -J train_phase15_mid_full
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_phase15_mid_full_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_phase15_mid_full_%J.err

#   bsub < jobs/train_phase15_mid_full.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_phase15_mid_sf18.pt"
EXTRA="${BLACKHOLE}/dataset_special_full.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/feb_sf/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/phase15_mid_full"

echo "=== Phase 1.5: mid_full ==="
echo "Dataset:    ${DATASET}"
echo "Extra:      ${EXTRA}"
echo "Init model: ${INIT_MODEL}"
echo "Out:        ${OUT_DIR}"
echo

for F in "${DATASET}" "${EXTRA}" "${INIT_MODEL}"; do
    if [ ! -f "${F}" ]; then echo "ERROR: not found: ${F}"; exit 1; fi
done

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

python3 -u "${SRC}/train.py" \
    --dataset        "${DATASET}" \
    --extra-dataset  "${EXTRA}" \
    --init-model     "${INIT_MODEL}" \
    --out            "${OUT_DIR}" \
    --policy-weight  0.0 \
    --rank-reg       0.1 \
    --draw-reg       0.0 \
    --weight-decay   5e-4 \
    --lr             3e-4 \
    --epochs         30 \
    --geo-patience   5 \
    --num-workers    0

echo
echo "Done. Checkpoint: ${OUT_DIR}/best.pt"
