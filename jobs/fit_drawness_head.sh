#!/bin/bash
#BSUB -J fit_drawness_head
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/fit_drawness_head_%J.out
#BSUB -e /zhome/81/b/206091/logs/fit_drawness_head_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# ── Fit drawness head via sklearn logistic regression ─────────────────────────
#
# Adam-based training (train_drawness_bootstrap.sh) stalled: balanced positions
# converge to 0 (gate <0.3 passed) but KR vs KR plateaus at 0.502 (gate >0.7).
# DrawL stuck at 0.585 from epoch 6 onward — Adam found a suboptimal basin.
#
# Fix: sklearn L-BFGS finds the global BCE optimum in one shot.
# The probe (job 28320925) confirmed 99% linear separability (Cohen's d=3.769)
# on the frozen phase15_mid_no_endgame geometry — the hyperplane exists.
#
# Generates 50k draw positions (stages 9/10/11) + 50k decisive (stages 1/2/4/5),
# runs the frozen backbone to get 128-dim embeddings, fits LR, writes weights
# directly into drawness_head[0].weight and bias. No gradient descent.
#
# Success gate: KR vs KR drawness >0.7 AND Sicilian drawness <0.3

python3 -u /zhome/81/b/206091/Petra-ver2/src/fit_drawness_head.py \
    --model /zhome/81/b/206091/Petra-ver2/models/phase15_mid_no_endgame/best.pt \
    --out   /zhome/81/b/206091/Petra-ver2/models/drawness_bootstrap/best.pt \
    --n     50000 \
    --C     1.0
