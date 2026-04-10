#!/bin/bash
#BSUB -J train_sf_gpu
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_sf_gpu_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_sf_gpu_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.0

# Supervised training on SF-labeled Lichess data mixed with 15% endgame anchors.
#
# This is the correct Session 8 bootstrap training path:
#   primary dataset : data/dataset_sf.pt   (200k positions, SF depth 15)
#   anchor dataset  : data/endgame_anchor.pt (KQK + KRK, ~40k positions)
#   anchor fraction : 15% of train batch per epoch
#
# Prerequisites:
#   bsub < jobs/gen_endgame.sh  (generates data/endgame_anchor.pt)
#
# The endgame anchor keeps the win/loss geometry axis anchored during training
# on the noisier SF-labeled Lichess positions. Without this, the value head
# can learn correct scalar outputs without the bottleneck developing geometric
# structure (the passenger problem documented in Phase 1 R1-R4).
#
# Output: models/best.pt  (replaces the current model trained without anchor)

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset        /zhome/81/b/206091/Petra-ver2/data/dataset_sf.pt \
    --anchor-dataset /zhome/81/b/206091/Petra-ver2/data/endgame_anchor.pt \
    --anchor-frac    0.15 \
    --out            /zhome/81/b/206091/Petra-ver2/models/ \
    --epochs         20 \
    --lr             5e-4 \
    --patience       5
