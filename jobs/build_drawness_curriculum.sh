#!/bin/bash
set -euo pipefail
#BSUB -J build_drawness_curriculum
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/build_drawness_curriculum_%J.out
#BSUB -e /zhome/81/b/206091/logs/build_drawness_curriculum_%J.err

# Build focused drawness curriculum from ELO 2000/2100/2200 datasets.
# Extracts all drawness positives (~76k) + 400k decisive positions.
#
#   bsub < jobs/build_drawness_curriculum.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

OUT="${BLACKHOLE}/dataset_drawness_curriculum.pt"

echo "=== Build drawness curriculum ==="
echo "Sources:"
echo "  ${BLACKHOLE}/dataset_elo2000_sf18.pt"
echo "  ${BLACKHOLE}/dataset_elo2100_sf18.pt"
echo "  ${BLACKHOLE}/dataset_elo2200_sf18.pt"
echo "Out: ${OUT}"
echo

for F in \
    "${BLACKHOLE}/dataset_elo2000_sf18.pt" \
    "${BLACKHOLE}/dataset_elo2100_sf18.pt" \
    "${BLACKHOLE}/dataset_elo2200_sf18.pt"; do
    if [ ! -f "${F}" ]; then echo "ERROR: not found: ${F}"; exit 1; fi
done

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/build_drawness_curriculum.py" \
    --datasets \
        "${BLACKHOLE}/dataset_elo2000_sf18.pt" \
        "${BLACKHOLE}/dataset_elo2100_sf18.pt" \
        "${BLACKHOLE}/dataset_elo2200_sf18.pt" \
    --out                 "${OUT}" \
    --n-decisive          400000 \
    --decisive-threshold  0.5 \
    --val-frac            0.05 \
    --seed                42

echo
echo "Done. Dataset: ${OUT}"
echo "Next: submit draw-reg ablation training on curriculum dataset"
