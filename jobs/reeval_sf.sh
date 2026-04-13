#!/bin/bash
#BSUB -J reeval_sf
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 12:00
#BSUB -o /zhome/81/b/206091/logs/reeval_sf_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_sf_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

python3 -u /zhome/81/b/206091/Petra-ver2/src/reeval_stockfish.py \
    --dataset /zhome/81/b/206091/Petra-ver2/data/dataset.pt \
    --out /zhome/81/b/206091/Petra-ver2/data/dataset_sf.pt \
    --stockfish /zhome/81/b/206091/bin/stockfish \
    --depth 15 \
    --n 200000 \
    --workers 16 \
    --min-decisive 0.3 \
    --max-pieces 20
