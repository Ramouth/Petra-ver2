#!/bin/bash
#BSUB -J train_2023_03_endgame
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 8:00
#BSUB -o /zhome/81/b/206091/logs/train_2023_03_endgame_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_2023_03_endgame_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# Training on Lichess 2023-03 (800k, SF depth-18) + endgame pool (SF depth-15).
#
# Key design:
#   - Primary dataset: 800k Lichess positions, SF depth-18 labels
#   - Anchor:          endgame_sf15.pt (956k positions, 45.8% near-zero draws)
#   - anchor-frac 0.25 → ~200k endgame positions mixed in per epoch
#   - draw-reg 0.02 → train auxiliary structural-draw head:
#       structural endgame draws as positives, decisive positions as negatives
#   - Total training set: ~1M positions per epoch
#   - Endgame pool replaces the feb_sf anchor: KR vs KR, KNN vs K, KB vs KB
#     stages ensure the draw dimension stays open
#
# Stopping: geometry patience (rank + win·draw cosine), not loss patience.
#   best.pt saved when geometry improves, training stops after 3 epochs
#   without rank or cosine improvement.
#
# Gates (run probe_geometry_2023_03_endgame.sh after training):
#   rank        > 18.9  (feb_sf baseline)
#   win·draw cos < 0.1676 (feb_sf baseline)
#   KR vs KR sanity check: |value| < 0.35
#   drawness: KR vs KR > 0.7, sharp balanced middlegame < 0.3

BLACKHOLE="/dtu/blackhole/0b/206091"

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset        ${BLACKHOLE}/dataset_2023_03_sf18.pt \
    --anchor-dataset ${BLACKHOLE}/endgame_sf15.pt \
    --anchor-frac    0.25 \
    --init-model     /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --rank-reg       0.1 \
    --draw-reg       0.02 \
    --num-workers    0 \
    --weight-decay   5e-4 \
    --lr             5e-4 \
    --epochs         20 \
    --geo-patience   3 \
    --out            /zhome/81/b/206091/Petra-ver2/models/lichess_2023_03_endgame
