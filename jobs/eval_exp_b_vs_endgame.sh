#!/bin/bash
#BSUB -J eval_exp_b_vs_endgame
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_exp_b_vs_endgame_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_exp_b_vs_endgame_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Head-to-head: exp_b (no policy, rank 72.9) vs lichess_2023_03_endgame (rank 21.6).

echo "=== exp_b (no policy, rank 72.9) vs lichess_2023_03_endgame (rank 21.6) ==="
python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model          /zhome/81/b/206091/Petra-ver2/models/exp_b_no_policy_strong/best.pt \
    --baseline-model /zhome/81/b/206091/Petra-ver2/models/lichess_2023_03_endgame/best.pt \
    --step 5 \
    --n-sim 50 \
    --games 100 \
    --workers 16
