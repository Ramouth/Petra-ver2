#!/bin/bash
set -euo pipefail
#BSUB -J train_policy_dr03_init
#BSUB -q gpul40s
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/train_policy_dr03_init_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_policy_dr03_init_%J.err

# Phase 2 §2 — frozen-trunk policy training on dr03.
#
# Background:
#   probe_policy.py (2026-05-02) confirmed both natural and dr03 policy heads
#   are statistically random init (illegal_mass p50 ≈ 0.993, masked normalised
#   entropy p50 ≈ 0.98). The masking fix in train.py (line 495) was added
#   18 days ago but neither champion's training run actually exercised it,
#   so both heads received no useful policy gradient.
#
#   The data side is already correct in dataset_drawness_curriculum.pt:
#     - move_idxs = SF best move per position (set by reeval_stockfish.py)
#     - legal_masks = 4096-bit packed legal-move mask (free, already cached)
#
# What this job does:
#   - Loads dr03's best.pt (geometry-good, policy-noise).
#   - --cold-init-policy: discards policy_head weights, fresh kaiming init.
#   - --freeze-trunk-train-policy: only the 528k-param policy_head receives
#     gradients. Value head, drawness head, and the entire trunk stay frozen.
#   - Trains policy with masked CE on existing SF-best targets + cached
#     legal masks. Same data dr03 was trained on — no re-reeval needed.
#
# Sanity gate (re-run probe_policy.py against the new best.pt):
#   illegal_mass p50      < 0.05    — masking is doing its job
#   masked norm-entropy p50 < 0.5   — head is contextual, not flat
#   per-phase consistent
#
# If both pass, §2 is done and the (dr03 trunk + new policy head) pair is
# the substrate for §3 (ELO bracket with opening book) and any later
# self-play.
#
# If illegal_mass crashes but entropy stays high (head learned legality
# but not preference), v2 adds light label smoothing or extends the
# reeval to multi-PV soft targets.
#
#   bsub < jobs/train_policy_dr03_init.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

DATASET="${BLACKHOLE}/dataset_drawness_curriculum.pt"
INIT_MODEL="${HOME_DIR}/Petra-ver2/models/drawness_full_natural_init_dr03/best.pt"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/policy_dr03_init"

echo "=== Phase 2 §2 — frozen-trunk policy training on dr03 ==="
echo "Dataset:    ${DATASET}"
echo "Init model: ${INIT_MODEL}"
echo "Out:        ${OUT_DIR}"
echo "Recipe:     freeze trunk + cold-init policy_head, masked CE on SF-best targets"
echo

for F in "${DATASET}" "${INIT_MODEL}"; do
    if [ ! -f "${F}" ]; then echo "ERROR: not found: ${F}"; exit 1; fi
done

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

python3 -u "${SRC}/train.py" \
    --dataset                   "${DATASET}" \
    --init-model                "${INIT_MODEL}" \
    --out                       "${OUT_DIR}" \
    --freeze-trunk-train-policy \
    --cold-init-policy \
    --policy-weight             1.0 \
    --weight-decay              1e-4 \
    --lr                        3e-4 \
    --epochs                    15 \
    --geo-patience              5 \
    --num-workers               0

echo
echo "Done. Model: ${OUT_DIR}/best.pt"
echo
echo "Next: re-run probe_policy.py against ${OUT_DIR}/best.pt to verify"
echo "      illegal_mass p50 < 0.05 and masked norm-entropy p50 < 0.5."
