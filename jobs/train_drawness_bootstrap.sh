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
# Strategy: fine-tune from lichess_2023_03_endgame (rank 21.6, first KR vs KR
# pass) on a
# mixed endgame set — decisive stages 1/2/4/5 as drawness negatives,
# drawn stages 9/10/11 as drawness positives.  policy_weight=0 keeps
# policy head weights frozen (no policy gradient) and also avoids
# materialising visit_dists tensors (~16 GB), making 250k positions
# feasible in 8 GB RAM (train.py skips visit_dists when policy_weight=0).
#
# Only 5 epochs to limit geometry regression from value-only endgame training.
# rank_reg=0.05 resists eigenvalue concentration while the draw dimension opens.
#
# Success gate (run probe_geometry.sh after training):
#   - drawness sanity: KR vs KR > 0.7, Sicilian < 0.3
#   - Effective rank not lower than feb_sf baseline (18.9) by more than 3
#   - All value sign checks pass
#
# After this job: run train_post_bootstrap.sh (broad SF/Lichess with low
# draw-reg anchor) to restore policy and broaden the geometry.

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --endgame-positions 250000 \
    --endgame-stages    1 2 4 5 9 10 11 \
    --init-model   /zhome/81/b/206091/Petra-ver2/models/lichess_2023_03_endgame/best.pt \
    --out          /zhome/81/b/206091/Petra-ver2/models/drawness_bootstrap \
    --draw-reg     0.05 \
    --rank-reg     0.05 \
    --policy-weight 0.0 \
    --epochs       5 \
    --geo-patience 5
