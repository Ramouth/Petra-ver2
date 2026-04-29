#!/bin/bash
#BSUB -J eval_drawness_head_v2_vs_feb_sf
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_drawness_head_v2_vs_feb_sf_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_drawness_head_v2_vs_feb_sf_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Head-to-head: drawness_head_v2 (rank 75.6, KR vs KR drawness=0.714, frozen
# backbone with LR-initialised head) vs feb_sf (rank 18.9, no drawness head).
# Gate: win rate > 50% confirms frozen+LR-init drawness training is net positive
# at no rank cost — strictly better than the previous champion.

echo "=== drawness_head_v2 vs feb_sf (head-to-head) ==="
python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model          /zhome/81/b/206091/Petra-ver2/models/drawness_head_v2/best.pt \
    --baseline-model /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --step 5 \
    --n-sim 100 \
    --games 200 \
    --workers 16
