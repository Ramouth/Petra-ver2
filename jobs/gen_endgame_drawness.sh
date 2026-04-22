#!/bin/bash
#BSUB -J gen_endgame_drawness
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/gen_endgame_drawness_%J.out
#BSUB -e /zhome/81/b/206091/logs/gen_endgame_drawness_%J.err

source /zhome/81/b/206091/petra-env/bin/activate

# Generate endgame anchor with explicit drawness supervision.
#
# Stages 1, 2, 4, 5 — decisive (drawness_target=0.0, drawness_mask=True)
# Stages 9, 10, 11  — structural draws (drawness_target=1.0, drawness_mask=True)
#
# This anchor is used with --draw-reg during regular Lichess/SF training.
# The drawness head learns directly from the explicit targets while the
# primary Lichess positions maintain geometry rank — no separate bootstrap
# job needed.
#
# With 30k base positions across 6 stages + mirrors: ~120k total positions.
# At anchor-frac=0.15 against a 400k Lichess dataset: ~60k anchor rows mixed in.

python3 -u /zhome/81/b/206091/Petra-ver2/src/generate_endgame.py \
    --positions 30000 \
    --stages 1 2 4 5 9 10 11 \
    --out /zhome/81/b/206091/Petra-ver2/data/endgame_drawness.pt \
    --seed 42
