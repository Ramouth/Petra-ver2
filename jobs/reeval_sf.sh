#!/bin/bash
#BSUB -J reeval_sf
#BSUB -q hpc
#BSUB -n 32
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 6:00
#BSUB -o /zhome/81/b/206091/logs/reeval_sf_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_sf_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

python3 /zhome/81/b/206091/Petra-ver2/src/reeval_stockfish.py \
    --dataset /zhome/81/b/206091/Petra-ver2/data/dataset.pt \
    --out /zhome/81/b/206091/Petra-ver2/data/dataset_sf.pt \
    --stockfish /zhome/81/b/206091/bin/stockfish \
    --depth 15 \
    --n 200000 \
    --workers 32
