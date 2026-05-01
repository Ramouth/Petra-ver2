#!/bin/bash
set -euo pipefail
#BSUB -J train_big_2025
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/train_big_2025_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_big_2025_%J.err

# Train `big_2025`: outcome-supervised model on the FULL 6.35M positions
# from 2025-02 + 2025-04 merged datasets at min_elo=2400 (~10% drawn,
# zero overlap with natural/natural_v2 training data).
#
# Differs from the existing train_big.sh:
#   - Skips the build_natural_v4 downsample (which targets 5M @ 7.5% drawn).
#     Uses both merged datasets directly via --dataset + --extra-dataset,
#     keeping all 6.35M positions and the full 10% draw rate.
#     Rationale: drawness PoC is the current primary focus; more drawn
#     positions in the corpus = more signal for the geometry to organise.
#   - Init from natural/best.pt — current ELO baseline; the natural recipe
#     reshaped 2021_06_all to a draw-aware geometry, so it's the right
#     starting point for further data ablation on this lineage.
#   - No drawness scaffolding (no BCE, no draw-reg). Per the
#     "drawness is a data problem" memory, the data delivers the signal;
#     scaffolding hasn't paid for its complexity in past runs.
#   - policy-weight 1.0 keeps the playing strength alive (ELO is secondary
#     but not abandoned per project_current_focus).
#
# Values in dataset_2025_{02,04}.pt are STM-relative game outcomes
# {-1, 0, +1} from the raw Lichess parse — no SF reeval, no outcome
# substitution needed. This is plain outcome supervision.
#
# Submit:
#   bsub < jobs/train_big_2025.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_2025_02.pt"
EXTRA="${BLACKHOLE}/dataset_2025_04.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/natural/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/big_2025"

echo "=== Train big_2025 (outcome supervision, 6.35M @ ~10% drawn) ==="
echo "Dataset:    ${DATASET}"
echo "Extra:      ${EXTRA}"
echo "Init model: ${INIT_MODEL}"
echo "Out:        ${OUT_DIR}"
echo "Recipe:     rank-reg 0.5, draw-reg 0.0, policy-weight 1.0, lr 3e-4, bs 512, 25ep, geo-patience 10"
echo

for F in "${DATASET}" "${EXTRA}" "${INIT_MODEL}"; do
    if [ ! -f "${F}" ]; then echo "ERROR: not found: ${F}"; exit 1; fi
done

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

python3 -u "${SRC}/train.py" \
    --dataset       "${DATASET}" \
    --extra-dataset "${EXTRA}" \
    --init-model    "${INIT_MODEL}" \
    --out           "${OUT_DIR}" \
    --policy-weight 1.0 \
    --rank-reg      0.5 \
    --draw-reg      0.0 \
    --weight-decay  5e-4 \
    --lr            3e-4 \
    --batch-size    512 \
    --epochs        25 \
    --geo-patience  10

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
echo
echo "Next steps:"
echo "  bsub -env \"VARIANT=big_2025\" < jobs/probe_drawness_head.sh    # geometry health check"
echo "  bsub < jobs/poc_battery_eval.sh                                  # add big_2025 to gate"
echo "  bsub < jobs/eval_natural_vs_feb_sf.sh                            # honest ELO gate (edit MODEL=big_2025)"
