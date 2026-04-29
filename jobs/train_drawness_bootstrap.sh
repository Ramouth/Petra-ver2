#!/bin/bash
#BSUB -J drawness_bootstrap
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/drawness_bootstrap_%J.out
#BSUB -e /zhome/81/b/206091/logs/drawness_bootstrap_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# ── Drawness bootstrap (v2) ────────────────────────────────────────────────────
#
# Goal: teach the auxiliary drawness head to distinguish structural draws
# (KR vs KR, KNN vs K, KB vs KB) from decisive unequal positions, using
# explicit drawness_target labels emitted by generate_endgame.py.
#
# Strategy: freeze the backbone and train only the drawness_head (129 params)
# on frozen geometry from phase15_mid_no_endgame (rank 89.9).
#
# Probe (job 28320925) confirmed: the rank-89.9 geometry separates structural
# draws from balanced positions at 99% logistic regression accuracy, Cohen's d=3.769.
# The drawness head is simply untrained (constant ~0.53 output, --draw-reg 0.0
# during phase15 training). The hyperplane is already in the geometry.
# --freeze-backbone guarantees rank stays at 89.9.
#
# v1 failure (job 28320933): draw-reg=0.05 starved the drawness head — GNorm=0.002
# (20× too small). DrawL barely moved (0.641→0.639) in 2 epochs.  Gap only 0.075.
# Fix: draw-reg=1.0 (20× stronger gradient) + 10 epochs + geo-patience=10.
# best.pt saving also fixed to trigger on draw_improved (not just rank_improved)
# since frozen backbone means rank on endgame val won't move after epoch 1.
#
# Success gate (run probe_geometry.sh after training):
#   - drawness sanity: KR vs KR > 0.7, Sicilian < 0.3
#   - Effective rank unchanged from init (89.9) — guaranteed by frozen backbone

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --endgame-positions 250000 \
    --endgame-stages    1 2 4 5 9 10 11 \
    --init-model   /zhome/81/b/206091/Petra-ver2/models/phase15_mid_no_endgame/best.pt \
    --out          /zhome/81/b/206091/Petra-ver2/models/drawness_bootstrap \
    --draw-reg     1.0 \
    --policy-weight 0.0 \
    --freeze-backbone \
    --epochs       10 \
    --geo-patience 10
