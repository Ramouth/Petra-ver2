#!/bin/bash
#BSUB -J probe_draw_reg_${DRAW_REG:-0.0}
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/probe_draw_reg_ablation_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_draw_reg_ablation_%J.err

# Probe geometry for one draw-reg ablation condition.
# Submit after the corresponding train job completes:
#
#   bsub -env "DRAW_REG=0.0" < jobs/probe_draw_reg_ablation.sh
#   bsub -env "DRAW_REG=0.2" < jobs/probe_draw_reg_ablation.sh
#   bsub -env "DRAW_REG=0.5" < jobs/probe_draw_reg_ablation.sh
#   bsub -env "DRAW_REG=1.0" < jobs/probe_draw_reg_ablation.sh
#   bsub -env "DRAW_REG=2.0" < jobs/probe_draw_reg_ablation.sh
#
# Compare across conditions:
#   - Effective rank        (does draw-reg hurt geometry expansion?)
#   - W·D cosine            (does draw region open?)
#   - KR vs KR value        (does backbone learn structural draws?)
#   - Drawness gap          (structural vs balanced separation)

DRAW_REG="${DRAW_REG:-0.0}"

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

TAG="${DRAW_REG//./p}"
MODEL="${HOME_DIR}/Petra-ver2/models/draw_reg_abl/dr${TAG}/best.pt"
PROBE_DATASET="${HOME_DIR}/Petra-ver2/data/dataset_2021_06_mid_sf18.pt"

echo "=== Draw-reg ablation probe: DRAW_REG=${DRAW_REG} ==="
echo "Model:   ${MODEL}"
echo "Dataset: ${PROBE_DATASET}"
echo

if [ ! -f "${MODEL}" ]; then echo "ERROR: model not found: ${MODEL}"; exit 1; fi
if [ ! -f "${PROBE_DATASET}" ]; then echo "ERROR: dataset not found: ${PROBE_DATASET}"; exit 1; fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/probe_geometry.py" \
    --model   "${MODEL}" \
    --dataset "${PROBE_DATASET}" \
    --n       5000
