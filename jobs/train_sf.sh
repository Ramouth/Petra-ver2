#!/bin/bash
#BSUB -J train_sf
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_sf_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_sf_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset /zhome/81/b/206091/Petra-ver2/data/dataset_sf.pt \
    --out /zhome/81/b/206091/Petra-ver2/models/
