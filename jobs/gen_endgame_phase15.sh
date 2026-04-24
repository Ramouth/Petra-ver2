#!/bin/bash
#BSUB -J gen_endgame_phase15
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/gen_endgame_phase15_%J.out
#BSUB -e /zhome/81/b/206091/logs/gen_endgame_phase15_%J.err

# Phase 1.5 endgame dataset — all 11 stages with explicit drawness labels.
#
# Decisive (drawness_target=0.0):
#   1=KQvK  2=KRvK  3=KPvK  4=KQvKR  5=KRvKP  6=KBvKP  7=KNvKP  8=KPvKP
# Drawn (drawness_target=1.0):
#   9=KRvKR  10=KNNvK  11=KBvKB
#
# 25k base positions × 11 stages × ~4 mirror/turn variants ≈ 110k total.
# Used as --anchor-dataset with --draw-reg in Phase 1.5 training.
#
#   bsub < jobs/gen_endgame_phase15.sh

source /zhome/81/b/206091/petra-env/bin/activate

python3 -u /zhome/81/b/206091/Petra-ver2/src/generate_endgame.py \
    --positions     25000 \
    --stages        1 2 3 4 5 6 7 8 9 10 11 \
    --out           /zhome/81/b/206091/Petra-ver2/data/endgame_phase15.pt \
    --seed          42

echo
echo "Done: /zhome/81/b/206091/Petra-ver2/data/endgame_phase15.pt"
