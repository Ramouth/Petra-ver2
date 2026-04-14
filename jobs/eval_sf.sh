#!/bin/bash
#BSUB -J eval_sf
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_sf_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_sf_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model /zhome/81/b/206091/Petra-ver2/models/sf_balanced/best.pt \
    --games 100 \
    --step 2 \
    --workers 16

python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model /zhome/81/b/206091/Petra-ver2/models/sf_balanced/best.pt \
    --games 100 \
    --step 5 \
    --workers 16
