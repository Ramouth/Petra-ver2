#!/bin/bash
#BSUB -J eval_geo
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_geo_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_geo_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model /zhome/81/b/206091/Petra-ver2/models/best.pt \
    --step 6 \
    --games 100 \
    --workers 16 \
    --probe-dataset /zhome/81/b/206091/Petra-ver2/data/dataset.pt \
    --pgn-out /zhome/81/b/206091/Petra-ver2/data/eval_geo.pgn
