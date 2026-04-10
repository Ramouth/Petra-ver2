#!/bin/bash
#BSUB -J gen_endgame
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 0:15
#BSUB -o /zhome/81/b/206091/logs/gen_endgame_%J.out
#BSUB -e /zhome/81/b/206091/logs/gen_endgame_%J.err

source /zhome/81/b/206091/petra-env/bin/activate

# Generate endgame anchor dataset for use as --sf-anchor in zigzag.
#
# Stages 1+2: KQ vs K and KR vs K.
# These positions have unambiguous labels (+1/-1) and antipodal mirrors
# are included automatically (so win/loss geometry is perfectly balanced).
# With --positions 10000 and mirrors=True, expect ~40k total positions.
#
# Why this matters: the original Petra (Phase 1) showed geometry stagnated
# across 4 zigzag rounds. The passenger problem: the value head learns to
# output correct scalars without the geometry needing to be structured.
# Endgame positions force the geometry — the labels are too clean to
# compensate around. Mixing 20% of these into every training round keeps
# the win/loss axis anchored even as self-play noise accumulates.

python3 -u /zhome/81/b/206091/Petra-ver2/src/generate_endgame.py \
    --positions 10000 \
    --stages 1 2 \
    --out /zhome/81/b/206091/Petra-ver2/data/endgame_anchor.pt \
    --seed 42
