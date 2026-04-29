#!/bin/bash
set -euo pipefail
#BSUB -J probe_drawness_${VARIANT:-head}
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/probe_drawness_head_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_drawness_head_%J.err

# Probe a drawness model. Set VARIANT to select which model:
#   bsub -env "VARIANT=head" < jobs/probe_drawness_head.sh   # frozen backbone
#   bsub -env "VARIANT=full" < jobs/probe_drawness_head.sh   # unfrozen backbone

VARIANT="${VARIANT:-head}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

MODEL="${HOME_DIR}/Petra-ver2/models/drawness_${VARIANT}/best.pt"
PROBE_DATASET="${BLACKHOLE}/dataset_2021_06_mid_sf18.pt"

echo "=== Drawness head probe ==="
echo "Model:   ${MODEL}"
echo "Dataset: ${PROBE_DATASET}"
echo

for F in "${MODEL}" "${PROBE_DATASET}"; do
    if [ ! -f "${F}" ]; then echo "ERROR: not found: ${F}"; exit 1; fi
done

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/probe_geometry.py" \
    --model   "${MODEL}" \
    --dataset "${PROBE_DATASET}" \
    --n       5000

echo
echo "=== Done: probe_drawness_head ==="
