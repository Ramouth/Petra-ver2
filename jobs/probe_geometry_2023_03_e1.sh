#!/bin/bash
#BSUB -J probe_geometry_2023_03_e1
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_geometry_2023_03_e1_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_geometry_2023_03_e1_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Geometry probe for lichess_2023_03 1-epoch model.
#
# Baselines:
#   feb_sf: rank=18.9, win·draw cosine (strict)=0.1676
#
# Gates:
#   rank > 18.9  (any improvement over feb_sf)
#   win·draw cosine < 0.1676

BLACKHOLE="/dtu/blackhole/0b/206091"

python3 -u /zhome/81/b/206091/Petra-ver2/src/probe_geometry.py \
    --model   /zhome/81/b/206091/Petra-ver2/models/lichess_2023_03_e1/best.pt \
    --dataset ${BLACKHOLE}/dataset_2023_03_sf18.pt \
    --n 5000
