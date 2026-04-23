#!/bin/bash
#BSUB -J train_2021_06_low_mid
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/train_2021_06_low_mid_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_2021_06_low_mid_%J.err

# Exp D (roadmap): no policy, low + mid bands (1200-2300 ELO).
# Does low-ELO decisiveness add geometry signal vs mid-only?

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_2021_06_mid_sf18.pt"
EXTRA_DATASET="${BLACKHOLE}/dataset_2021_06_low_sf18.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/exp_b_no_policy_strong/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/2021_06_low_mid"

echo "=== Exp D: no policy, low + mid (1200-2300) ==="
echo "Dataset:       ${DATASET}"
echo "Extra dataset: ${EXTRA_DATASET}"
echo "Init model:    ${INIT_MODEL}"
echo "Out:           ${OUT_DIR}"
echo

for F in "${DATASET}" "${EXTRA_DATASET}" "${INIT_MODEL}"; do
    if [ ! -f "${F}" ]; then echo "ERROR: not found: ${F}"; exit 1; fi
done

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

python3 -u "${SRC}/train.py" \
    --dataset        "${DATASET}" \
    --extra-dataset  "${EXTRA_DATASET}" \
    --init-model     "${INIT_MODEL}" \
    --out            "${OUT_DIR}" \
    --policy-weight  0.0 \
    --rank-reg       0.5 \
    --draw-reg       0.0 \
    --weight-decay   5e-4 \
    --lr             3e-4 \
    --epochs         30 \
    --geo-patience   5 \
    --num-workers    0

echo
echo "Done. Checkpoint: ${OUT_DIR}/best.pt"
echo "Next: bsub -env \"EXP=low_mid\" < jobs/eval_2021_06_exp.sh"
