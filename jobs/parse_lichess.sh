#!/bin/bash
#BSUB -J parse_lichess
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/parse_lichess_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_lichess_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
pip install zstandard --quiet

python3 -u /zhome/81/b/206091/Petra-ver2/src/data.py \
    --pgn /zhome/81/b/206091/Petra-ver2/data/lichess_db_standard_rated_2020-03.pgn.zst \
    --out /zhome/81/b/206091/Petra-ver2/data/dataset.pt \
    --max-games 150000 \
    --min-elo 1500
