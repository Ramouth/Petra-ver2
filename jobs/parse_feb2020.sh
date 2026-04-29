#!/bin/bash
#BSUB -J parse_feb2020
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/parse_feb2020_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_feb2020_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
pip install zstandard --quiet

# Parse February 2020 Lichess database with versatile position sampling.
#
# Changes vs parse_lichess.sh (January 2020):
#   --positions-per-game 20   (was 8) — 2.5x more coverage per game
#   --skip-opening 5          (was 10) — include early middlegame
#   --sampling even           (was random) — one position from each game phase
#                             (opening, early/mid/late middlegame, endgame)
#                             instead of clustered random samples
#
# Expected output: ~150k × 20 = ~3M positions → ~3.5GB .pt file
# After this completes, delete the PGN to free ~18GB:
#   rm ~/Petra-ver2/data/lichess_db_standard_rated_2020-02.pgn.zst

python3 -u /zhome/81/b/206091/Petra-ver2/src/data.py \
    --pgn               /zhome/81/b/206091/Petra-ver2/data/lichess_db_standard_rated_2020-02.pgn.zst \
    --out               /zhome/81/b/206091/Petra-ver2/data/dataset_feb.pt \
    --max-games         150000 \
    --min-elo           1500 \
    --positions-per-game 20 \
    --skip-opening      5 \
    --sampling          even \
    --no-strict
