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

# Baseline geometry health check on the supervised SF model.
# Run this before zigzag round 1 to get baseline numbers:
#   - Effective rank (target > 30)
#   - Win/loss centroid cosine (lower = better; R4 was 0.869)
#   - Separation gap (higher = better; R4 was 0.048, R2 peak was 0.057)
#   - NN label consistency
# After each zigzag round, rerun with the new model to track progress.
# If separation gap does not improve after round 2, consider endgame anchoring.

python3 -u /zhome/81/b/206091/Petra-ver2/src/probe_geometry.py \
    --model   /zhome/81/b/206091/Petra-ver2/models/best.pt \
    --dataset /zhome/81/b/206091/Petra-ver2/data/dataset_sf.pt \
    --n 5000
