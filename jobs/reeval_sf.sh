#!/bin/bash
#BSUB -J reeval_sf
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 8:00
#BSUB -o /zhome/81/b/206091/logs/reeval_sf_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_sf_%J.err

source /zhome/81/b/206091/petra-env/bin/activate

python3 /zhome/81/b/206091/Petra-ver2/src/reeval_stockfish.py \
    --dataset /zhome/81/b/206091/Petra-ver2/data/dataset.pt \
    --out /zhome/81/b/206091/Petra-ver2/data/dataset_sf.pt \
    --stockfish /zhome/81/b/206091/bin/stockfish \
    --depth 10 \
    --n 200000
