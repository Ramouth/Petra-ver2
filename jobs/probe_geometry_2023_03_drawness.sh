#!/bin/bash
#BSUB -J probe_geo_drawness
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_geo_drawness_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_geo_drawness_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Geometry probe for lichess_2023_03_drawness model.
#
# Gates (vs lichess_2023_03_endgame baseline: rank=21.6, wdcos=0.2128):
#   rank >= 21.6          (no regression from init)
#   win·draw cosine < 0.5 (not collapsed)
#   KR vs KR |value| < 0.35
#   drawness: KR vs KR > 0.7, sharp balanced < 0.3

BLACKHOLE="/dtu/blackhole/0b/206091"

python3 -u /zhome/81/b/206091/Petra-ver2/src/probe_geometry.py \
    --model   /zhome/81/b/206091/Petra-ver2/models/lichess_2023_03_drawness/best.pt \
    --dataset ${BLACKHOLE}/dataset_2023_03_sf18.pt \
    --n 5000
