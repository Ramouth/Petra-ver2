#!/bin/bash
#BSUB -J eval_natural_vs_2021_06_all
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_natural_vs_2021_06_all_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_natural_vs_2021_06_all_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Head-to-head: natural (curriculum-retrained, rank 33.8) vs 2021_06_all
# (its own init, rank 87.0). Same starting weights, both have policy —
# the only difference is whether the model went through curriculum
# training with rank-reg.
#
# Gate: did the curriculum trade help or hurt play strength?
#   wr > 50%: curriculum improved the model
#   wr ~ 50%: trade was neutral
#   wr < 50%: rank loss outweighed any geometry benefit

echo "=== natural vs 2021_06_all (head-to-head, same lineage) ==="
python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model          /zhome/81/b/206091/Petra-ver2/models/natural/best.pt \
    --baseline-model /zhome/81/b/206091/Petra-ver2/models/2021_06_all/best.pt \
    --step 5 \
    --n-sim 100 \
    --games 200 \
    --workers 16
