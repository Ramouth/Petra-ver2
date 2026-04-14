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
module load cuda/12.1

# Geometry health check on the balanced supervised model (sf_balanced).
# Baseline from previous sf_gpu run:
#   - Effective rank: 3.7 / 128  (collapsed)
#   - Separation gap: low
# Target after balanced training:
#   - Effective rank > 30
#   - Separation gap > 0.05
#   - β1 > 0 (topology healthy)

python3 -u /zhome/81/b/206091/Petra-ver2/src/probe_geometry.py \
    --model   /zhome/81/b/206091/Petra-ver2/models/sf_balanced/best.pt \
    --dataset /zhome/81/b/206091/Petra-ver2/data/dataset_balanced.pt \
    --n 5000
