#!/bin/bash
#BSUB -J train_sf_gpu
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=5GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_sf_gpu_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_sf_gpu_%J.err

set -x
echo "Activating virtualenv..."
source /zhome/81/b/206091/petra-env/bin/activate
echo "Python: $(which python3)"
module load gcc/13.4.0-binutils-2.44
module load cuda/11.8
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python3 -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# Current doover run:
#   primary signal = broad SF-labeled Lichess positions
#   anchor signal  = stage 1+2 endgame dataset (KQK + KRK)
# Goal: expand geometry rank while keeping a stable win/loss axis.

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset /zhome/81/b/206091/Petra-ver2/data/dataset_sf.pt \
    --anchor-dataset /zhome/81/b/206091/Petra-ver2/data/endgame_anchor.pt \
    --anchor-frac 0.15 \
    --out /zhome/81/b/206091/Petra-ver2/models/sf_gpu
