#!/bin/bash
#BSUB -J probe_geometry_r1
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/probe_geometry_r1_%J.out
#BSUB -e /zhome/81/b/206091/logs/probe_geometry_r1_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Post-round 1 geometry probe. Run after zigzag_r1.sh completes.
# Compare output against the baseline from probe_geometry.sh.
#
# Key numbers to watch:
#   Effective rank     — target > 30 (baseline from best.pt)
#   Separation gap     — must be HIGHER than baseline (R4 historical: 0.048)
#   Centroid cosine    — must be LOWER than baseline (R4 historical: 0.869)
#   KQ vs K (both colors) — must both classify correctly
#
# If separation gap does not improve vs baseline after round 1:
#   The passenger problem is active. Consider:
#   1. Increase --anchor-frac to 0.30 for round 2
#   2. Run gen_endgame.sh with --stages 1 2 3 4 for a richer endgame anchor
#   3. Consider explicit geometry loss (LATER.md: Fisher discriminant)

python3 -u /zhome/81/b/206091/Petra-ver2/src/probe_geometry.py \
    --model   /zhome/81/b/206091/Petra-ver2/models/zigzag/r1/best.pt \
    --dataset /zhome/81/b/206091/Petra-ver2/data/selfplay_r1_sf.pt \
    --n 5000
