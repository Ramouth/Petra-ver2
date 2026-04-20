#!/bin/bash
#BSUB -J probe_geometry_2023_03
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_geometry_2023_03_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_geometry_2023_03_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# Geometry probe for lichess_2023_03 model (SF depth-18 labels, draw=0.0).
#
# Baseline (feb_sf):
#   - Effective rank: 18.9
#   - win·draw cosine (strict): 0.1676
#
# Gates to pass:
#   - Effective rank > 20  (target: > 30)
#   - win·draw cosine (strict) < 0.1676  (draw dimension opening)
#
# If rank is flat vs feb_sf despite clean SF labels → architecture ceiling,
# consider backbone rewrite.

BLACKHOLE="/dtu/blackhole/0b/206091"

python3 -u /zhome/81/b/206091/Petra-ver2/src/probe_geometry.py \
    --model   /zhome/81/b/206091/Petra-ver2/models/lichess_2023_03/best.pt \
    --dataset ${BLACKHOLE}/dataset_2023_03_sf18.pt \
    --n 5000
