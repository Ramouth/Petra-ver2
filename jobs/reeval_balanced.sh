#!/bin/bash
#BSUB -J reeval_balanced
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/reeval_balanced_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_balanced_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Second reeval pass: full-board positions with balanced labels.
#
# Why: the supervised model (sf_gpu/best.pt) has effective rank 3.7/128 —
# almost pure win/loss axis with no middle ground. It was trained on
# max_pieces=20 positions (92.3% decisive), so:
#   1. Opening positions (32 pieces) are fully out-of-distribution
#   2. The model has no concept of a draw or balanced position
#   3. Self-play from the opening would reinforce this bias
#
# Fix: include full-board positions (max_pieces=32) at min_decisive=0.05,
# which is above the SF noise floor (~20cp) and captures draws + slight
# advantages. Same depth 15 as dataset_sf.pt for label consistency.
#
# Output is mixed with dataset_sf.pt in train_balanced_gpu.sh.

python3 -u /zhome/81/b/206091/Petra-ver2/src/reeval_stockfish.py \
    --dataset /zhome/81/b/206091/Petra-ver2/data/dataset_jan.pt \
    --out     /zhome/81/b/206091/Petra-ver2/data/dataset_balanced.pt \
    --stockfish /zhome/81/b/206091/bin/stockfish \
    --depth 15 \
    --n 200000 \
    --workers 16 \
    --min-decisive 0.05 \
    --max-pieces 32
