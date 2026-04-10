#!/bin/bash
#BSUB -J zigzag_r1
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 10:00
#BSUB -o /zhome/81/b/206091/logs/zigzag_r1_%J.out
#BSUB -e /zhome/81/b/206091/logs/zigzag_r1_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Zigzag round 1: self-play → SF reeval → train → gate
#
# Prerequisites (run these first):
#   bsub < jobs/gen_endgame.sh      (generates data/endgame_anchor.pt, ~5 min)
#   bsub < jobs/probe_geometry.sh   (baseline probe before any self-play)
#
# Round 1 parameters (from ZIGZAG.md):
#   n_sim=40, SF depth=12, 500 games, lr=5e-4
#
# --sf-anchor: mixes 20% KQK+KRK endgame positions into every training epoch.
#   This is the key fix vs Phase 1 — prevents the passenger problem by keeping
#   the win/loss axis anchored on unambiguous positions throughout self-play.
#
# --n-seeds 3: trains 3 random seeds in parallel, selects best geometry via
#   select_geometry.py. Converts initialisation variance into an advantage.
#   Free on a 16-core node (3 × ~5 cores each during training).
#
# Gate: 200 games at n_sim=50 vs MCTS(material). >55% to advance.
#
# After this job completes, run:
#   bsub < jobs/probe_geometry_r1.sh  (to check geometry vs baseline)

python3 -u /zhome/81/b/206091/Petra-ver2/src/zigzag.py \
    --rounds      1 \
    --workers     16 \
    --seed-model  /zhome/81/b/206091/Petra-ver2/models/best.pt \
    --sf-anchor   /zhome/81/b/206091/Petra-ver2/data/endgame_anchor.pt \
    --anchor-frac 0.20 \
    --n-seeds     3
