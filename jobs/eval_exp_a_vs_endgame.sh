#!/bin/bash
#BSUB -J eval_exp_a_vs_endgame
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_exp_a_vs_endgame_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_exp_a_vs_endgame_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Head-to-head: exp_a (no policy, rank 40.5) vs lichess_2023_03_endgame (rank 21.6).
# Key question: does rank 40.5 + untrained policy beat rank 21.6 + trained policy?
# If yes: geometry quality dominates search quality at this MCTS budget.
# If no: policy guidance is still necessary even with better value estimates.

echo "=== exp_a (no policy, rank 40.5) vs lichess_2023_03_endgame (rank 21.6) ==="
python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model          /zhome/81/b/206091/Petra-ver2/models/exp_a_no_policy/best.pt \
    --baseline-model /zhome/81/b/206091/Petra-ver2/models/lichess_2023_03_endgame/best.pt \
    --step 5 \
    --n-sim 100 \
    --games 200 \
    --workers 16
