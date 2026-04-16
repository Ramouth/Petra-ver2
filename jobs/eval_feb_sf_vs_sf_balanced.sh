#!/bin/bash
#BSUB -J eval_feb_sf_vs_sf_balanced
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_feb_sf_vs_sf_balanced_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_feb_sf_vs_sf_balanced_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Apples-to-apples: feb_sf vs sf_balanced at the same n_sim.
# Step 5 with --baseline-model runs:
#   MCTS(feb_sf, learned, n=100) vs MCTS(sf_balanced, learned, n=100)
# Baseline: both scored ~57-58% vs material; this tells us if the
# draw-inclusive training (min_decisive=0.0) added real playing strength.

echo "=== Step 5: feb_sf vs sf_balanced (head-to-head) ==="
python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model          /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --baseline-model /zhome/81/b/206091/Petra-ver2/models/sf_balanced/best.pt \
    --step 5 \
    --n-sim 100 \
    --games 200 \
    --workers 16
