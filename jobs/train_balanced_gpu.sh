#!/bin/bash
#BSUB -J train_balanced_gpu
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_balanced_gpu_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_balanced_gpu_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# Merged training run:
#   primary   = dataset_sf.pt      (March 2020, decisive, max_pieces=20)
#   extra     = dataset_balanced.pt (January 2020, balanced, max_pieces=32)
#   anchor    = endgame_anchor.pt  (KQK + KRK, perfectly balanced)
#
# Why merge:
#   dataset_sf.pt gave effective rank 3.7/128 — model has no middle ground.
#   dataset_balanced.pt adds full-board + balanced positions (min_decisive=0.05)
#   to teach the model that draws exist and that opening positions are OOD.
#   Different Lichess months to reduce position overlap between datasets.
#
# Overfitting mitigations:
#   --weight-decay 5e-4  (up from 1e-4 — L2 pressure against memorisation)
#   Val sets from BOTH datasets merged — early stopping catches overfitting
#   on either distribution.

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset        /zhome/81/b/206091/Petra-ver2/data/dataset_sf.pt \
    --extra-dataset  /zhome/81/b/206091/Petra-ver2/data/dataset_balanced.pt \
    --anchor-dataset /zhome/81/b/206091/Petra-ver2/data/endgame_anchor.pt \
    --anchor-frac    0.15 \
    --weight-decay   5e-4 \
    --out            /zhome/81/b/206091/Petra-ver2/models/sf_balanced
