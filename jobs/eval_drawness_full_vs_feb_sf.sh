#!/bin/bash
#BSUB -J eval_drawness_full_vs_feb_sf
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_drawness_full_vs_feb_sf_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_drawness_full_vs_feb_sf_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Head-to-head: drawness_full (rank 34.2, drawness gap +0.778)
# vs feb_sf (rank 18.9, no drawness head).
# Gate: win rate > 50% confirms rank 34 + draw awareness beats rank 18.9.

echo "=== drawness_full vs feb_sf (head-to-head) ==="
python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model          /zhome/81/b/206091/Petra-ver2/models/drawness_full/best.pt \
    --baseline-model /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --step 5 \
    --n-sim 100 \
    --games 200 \
    --workers 16
