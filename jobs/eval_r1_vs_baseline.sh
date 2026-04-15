#!/bin/bash
#BSUB -J eval_r1_vs_baseline
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_r1_vs_baseline_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_r1_vs_baseline_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Apples-to-apples: r1 vs sf_balanced at the same n_sim.
# Step 5 with --baseline-model runs:
#   MCTS(r1, learned, n=100) vs MCTS(sf_balanced, learned, n=100)
# This is the first true head-to-head — all prior comparisons were
# against material MCTS at different n_sim values.

echo "=== Step 5: r1 vs sf_balanced (head-to-head) ==="
python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model          /zhome/81/b/206091/Petra-ver2/models/zigzag/r1/best.pt \
    --baseline-model /zhome/81/b/206091/Petra-ver2/models/sf_balanced/best.pt \
    --step 5 \
    --n-sim 100 \
    --games 200 \
    --workers 16

# Step 6: does r1 geometry alone beat material?
# If yes: self-play strengthened the representational signal.
# If no:  geometry is not yet directionally structured enough for search.

echo "=== Step 6: r1 geometry vs material ==="
python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model         /zhome/81/b/206091/Petra-ver2/models/zigzag/r1/best.pt \
    --step 6 \
    --n-sim 100 \
    --games 200 \
    --workers 16 \
    --probe-dataset /zhome/81/b/206091/Petra-ver2/data/selfplay_r1_sf.pt

# Step 6: same for sf_balanced as a baseline reference.
# Tells us whether the geometry signal existed before self-play.

echo "=== Step 6: sf_balanced geometry vs material ==="
python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model         /zhome/81/b/206091/Petra-ver2/models/sf_balanced/best.pt \
    --step 6 \
    --n-sim 100 \
    --games 200 \
    --workers 16 \
    --probe-dataset /zhome/81/b/206091/Petra-ver2/data/dataset_balanced.pt
