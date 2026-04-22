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

# ── Drawness bootstrap ─────────────────────────────────────────────────────────
#
# Goal: teach the auxiliary drawness head to distinguish structural draws
# (KR vs KR, KNN vs K, KB vs KB) from decisive unequal positions, using
# explicit drawness_target labels emitted by generate_endgame.py.
#
# Strategy: freeze the backbone and train only the drawness_head (129 params)
# on frozen geometry from lichess_2023_03_endgame (rank 21.6, first KR vs KR pass).
#
# Previous attempt (job 28258816) showed the drawness head converges in 1 epoch
# (gap=+0.956) but rank collapsed 21.6→14.3 due to endgame value gradient reshaping
# the geometry.  --freeze-backbone eliminates that regression entirely: only the
# 129-parameter linear head is updated.
#
# policy_weight=0 avoids allocating visit_dists (~16 GB) for positions that will
# never use them (train.py skips storage when policy_weight=0).
#
# Success gate (run probe_geometry.sh after training):
#   - drawness sanity: KR vs KR > 0.7, Sicilian < 0.3
#   - Effective rank unchanged from init (21.6) — guaranteed by frozen backbone
#   - All value sign checks pass (value head frozen, should be identical)
#
# After this job: run broad SF/Lichess training with --init-model pointing here
# and a small --draw-reg to keep the drawness head anchored as geometry broadens.

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --endgame-positions 250000 \
    --endgame-stages    1 2 4 5 9 10 11 \
    --init-model   /zhome/81/b/206091/Petra-ver2/models/lichess_2023_03_endgame/best.pt \
    --out          /zhome/81/b/206091/Petra-ver2/models/drawness_bootstrap \
    --draw-reg     0.05 \
    --policy-weight 0.0 \
    --freeze-backbone \
    --epochs       2 \
    --geo-patience 2
