#!/bin/bash
#BSUB -J draw_reg_${DRAW_REG:-0.0}
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/draw_reg_ablation_%J.out
#BSUB -e /zhome/81/b/206091/logs/draw_reg_ablation_%J.err

# Draw-reg ablation on ELO 2000 dataset.
# Isolates draw-reg's effect on geometry: does it expand the draw dimension,
# hurt rank, or do nothing beyond training the head?
#
# All conditions are identical except DRAW_REG. Submit all 5 in parallel:
#
#   bsub -env "DRAW_REG=0.0" < jobs/train_draw_reg_ablation.sh
#   bsub -env "DRAW_REG=0.2" < jobs/train_draw_reg_ablation.sh
#   bsub -env "DRAW_REG=0.5" < jobs/train_draw_reg_ablation.sh
#   bsub -env "DRAW_REG=1.0" < jobs/train_draw_reg_ablation.sh
#   bsub -env "DRAW_REG=2.0" < jobs/train_draw_reg_ablation.sh
#
# After all complete: bsub -env "DRAW_REG=<val>" < jobs/probe_draw_reg_ablation.sh
# Key metrics: rank, W·D cosine, KR vs KR value, drawness gap.
# If rank and W·D cosine are flat across conditions → draw-reg is not reshaping
# geometry and we stop pursuing it as a mechanism.

DRAW_REG="${DRAW_REG:-0.0}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

TAG="${DRAW_REG//./p}"
DATASET="${BLACKHOLE}/dataset_elo2000_sf18.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/phase15_mid_no_endgame/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/draw_reg_abl/dr${TAG}"

echo "=== Draw-reg ablation: DRAW_REG=${DRAW_REG} ==="
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
    --draw-reg      "${DRAW_REG}" \
    --weight-decay  5e-4 \
    --lr            3e-4 \
    --epochs        20 \
    --geo-patience  5 \
    --num-workers   0

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
echo "Next: bsub -env \"DRAW_REG=${DRAW_REG}\" < jobs/probe_draw_reg_ablation.sh"
