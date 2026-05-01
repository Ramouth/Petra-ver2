#!/bin/bash
set -euo pipefail
#BSUB -J train_big_2025_filtered
#BSUB -q gpul40s
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/train_big_2025_filtered_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_big_2025_filtered_%J.err

# Train `big_2025_filtered`: outcome-supervised on the piece-count-filtered
# 2025 corpus (pc <= 23). See project_outcome_draw_composition memory.
#
# Differs from train_big_2025.sh:
#   - Single merged & filtered input (build_2025_filtered.py output) —
#     no --extra-dataset.
#   - Lower LR (1e-4 vs 3e-4). The previous run from natural/best.pt at
#     3e-4 destroyed natural's draw-aware geometry in a single epoch
#     (rank 33.8 → 18, gnorm climbed). Outcome labels are noisier per
#     position than SF-value labels; gentler updates needed.
#   - Same recipe otherwise (rank-reg 0.5, no draw scaffolding,
#     policy-weight 1.0, init from natural/best.pt).
#
# Prerequisite: `bsub < jobs/build_2025_filtered.sh` must have completed
# and produced /dtu/blackhole/0b/206091/dataset_2025_filtered_pc23.pt.
#
# Submit:
#   bsub < jobs/train_big_2025_filtered.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_2025_filtered_pc23.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/natural/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/big_2025_filtered"

echo "=== Train big_2025_filtered (outcome supervision, pc<=23) ==="
echo "Dataset:    ${DATASET}"
echo "Init model: ${INIT_MODEL}"
echo "Out:        ${OUT_DIR}"
echo "Recipe:     rank-reg 0.5, draw-reg 0.0, policy-weight 1.0, lr 1e-4, bs 512, 25ep, geo-patience 10"
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
    --policy-weight 1.0 \
    --rank-reg      0.5 \
    --draw-reg      0.0 \
    --weight-decay  5e-4 \
    --lr            1e-4 \
    --batch-size    512 \
    --epochs        25 \
    --geo-patience  10

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
echo
echo "Next steps:"
echo "  bsub -env \"VARIANT=big_2025_filtered\" < jobs/probe_drawness_head.sh"
echo "  bsub < jobs/poc_battery_eval.sh"
echo "  Edit jobs/eval_natural_vs_feb_sf.sh MODEL=big_2025_filtered for ELO gate"
