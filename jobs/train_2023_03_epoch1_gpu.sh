#!/bin/bash
#BSUB -J train_2023_03_e1
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/train_2023_03_e1_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_2023_03_e1_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# 1-epoch probe: test whether KR vs KR is already broken after a single pass
# through the 800k dataset, or whether the drift is a multi-epoch accumulation.
#
# Hypothesis: the dataset is large enough that epoch 1 ≈ the total useful
# training of prior small-data runs. If KR vs KR passes here but fails at
# epoch 3+, we confirm the geometry saturates fast and the epoch budget should
# be capped at 1-2 for this dataset size.
#
# Output goes to a separate dir so feb_sf/best.pt and lichess_2023_03/best.pt
# are not overwritten.

BLACKHOLE="/dtu/blackhole/0b/206091"

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset      ${BLACKHOLE}/dataset_2023_03_sf18.pt \
    --init-model   /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --rank-reg     0.1 \
    --num-workers  0 \
    --weight-decay 5e-4 \
    --lr           5e-4 \
    --epochs       1 \
    --out          /zhome/81/b/206091/Petra-ver2/models/lichess_2023_03_e1
