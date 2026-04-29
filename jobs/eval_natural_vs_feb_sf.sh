#!/bin/bash
#BSUB -J eval_natural_vs_feb_sf
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_natural_vs_feb_sf_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_natural_vs_feb_sf_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Head-to-head: natural (curriculum data, no drawness scaffolding, policy
# trained) vs feb_sf. First fair MCTS comparison since natural also has
# a properly trained policy.
#
# Gate: win rate > 50% confirms the curriculum's value distribution alone
# is enough to shape draw-aware geometry — drawness scaffolding was
# unnecessary.

echo "=== natural vs feb_sf (head-to-head, both have policy) ==="
python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model          /zhome/81/b/206091/Petra-ver2/models/natural/best.pt \
    --baseline-model /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --step 5 \
    --n-sim 100 \
    --games 200 \
    --workers 16
