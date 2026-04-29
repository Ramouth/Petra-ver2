#!/bin/bash
#BSUB -J eval_natural_v2_vs_2021_06_all
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_natural_v2_vs_2021_06_all_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_natural_v2_vs_2021_06_all_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Same-lineage head-to-head: natural_v2 (2M, elo2100 + drawn-elo2200, SF
# labels, no drawness scaffolding) vs 2021_06_all (its init).
#
# Comparison point: natural at 70.6% wr (4 losses in 80) vs 2021_06_all.
# If natural_v2 matches or beats that, scale + cleaner data sources help.
# If natural_v2 ties or loses, the curated curriculum was actually better.

echo "=== natural_v2 vs 2021_06_all (head-to-head, same lineage) ==="
python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model          /zhome/81/b/206091/Petra-ver2/models/natural_v2/best.pt \
    --baseline-model /zhome/81/b/206091/Petra-ver2/models/2021_06_all/best.pt \
    --step 5 \
    --n-sim 100 \
    --games 200 \
    --workers 16
