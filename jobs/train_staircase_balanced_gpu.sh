#!/bin/bash
#BSUB -J train_staircase_balanced
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_staircase_balanced_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_staircase_balanced_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# ── Staircase Stage B: balanced/drawn positions ────────────────────────────────
#
# Goal: push the draw cluster into its own orthogonal dimension, making it
# distinct from both the win and loss clusters.
#
# Data: balanced_*.pt — positions with |SF_value| < 0.3, depth-20 labels,
#       high-ELO games (2200+).  These are genuinely contested equal positions
#       where the primary signal is that the position is NOT decided — the draw
#       dimension must activate.
#
# Init: staircase_decisive/best.pt — chain from Stage A.
#       The win/loss axis is already positioned; Stage B specialises the draw
#       subspace without destabilising it.
#
# Why lower lr (5e-5): balanced positions have smaller |value|, so the gradient
# signal per position is weaker.  A lower lr prevents the network from being
# pulled hard by the rare positions at the boundary.
#
# Why lower rank_reg (0.05): at this stage we want to *direct* the draw cluster
# to a specific region (orthogonal to win/loss), not just maximise raw rank.
# Excessive rank_reg with weak labels can scatter the geometry randomly.
#
# Geometry gate (run probe_geometry.sh after training):
#   - strict win·draw cosine < 0.10  (was 0.1676 at feb_sf)
#   - strict win·loss cosine remains negative (draw movement should not collapse)
#   - Effective rank > 25 (or ideally > 30)
#   - KR vs KR value head output ≈ 0 (draw dimension open → value near neutral)
#
# After this job: run probe_geometry.sh to verify, then consider self-play harvest.

# ── Data paths — edit to match your split output ───────────────────────────────
python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset      /zhome/81/b/206091/Petra-ver2/data/balanced_2020_03.pt \
    --init-model   /zhome/81/b/206091/Petra-ver2/models/staircase_decisive/best.pt \
    --out          /zhome/81/b/206091/Petra-ver2/models/staircase_balanced \
    --lr           5e-5 \
    --weight-decay 5e-4 \
    --rank-reg     0.05 \
    --epochs       20 \
    --patience     5 \
    --num-workers  4
