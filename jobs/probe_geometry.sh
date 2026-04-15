#!/bin/bash
#BSUB -J probe_geometry
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_geometry_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_geometry_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Geometry health check on feb_sf model.
# Baseline (round2 / sf_balanced):
#   - win·draw cosine: 0.2657  (draw not separated)
#   - Effective rank: ~7.2
# Target after feb_sf training:
#   - Effective rank > 15 (stretch: > 30)
#   - win·draw centroid cosine closer to 0 (draw dimension opening)

python3 -u /zhome/81/b/206091/Petra-ver2/src/probe_geometry.py \
    --model   /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --dataset /zhome/81/b/206091/Petra-ver2/data/dataset_feb_sf.pt \
    --n 5000
