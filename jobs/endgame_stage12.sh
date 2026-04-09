#!/bin/bash
#BSUB -J endgame_stage12
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/endgame_stage12_%J.out
#BSUB -e /zhome/81/b/206091/logs/endgame_stage12_%J.err

source /zhome/81/b/206091/petra-env/bin/activate

python3 /zhome/81/b/206091/Petra-ver2/src/train.py \
    --endgame-positions 20000 \
    --endgame-stages 1 2 \
    --policy-weight 0 \
    --epochs 20 \
    --out /zhome/81/b/206091/Petra-ver2/models/endgame/stage12/
