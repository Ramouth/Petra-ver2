#!/bin/bash
#BSUB -J reeval_feb2020
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/reeval_feb2020_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_feb2020_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# SF re-evaluation of February 2020 parsed dataset.
#
# Source: dataset_feb.pt — 2.97M positions, evenly sampled across game arc
#   (skip_opening=5, positions_per_game=20, sampling=even)
#   This covers opening, middlegame, and endgame positions in every game.
#
# Key difference from reeval_balanced.sh:
#   --min-decisive 0.0   keep ALL positions including draws and equal positions.
#   Previous runs used 0.05, which effectively excluded most draws.
#   This run preserves the full decisiveness spectrum to test whether the
#   draw dimension opens up in the geometry (win·draw cosine currently 0.2657).
#
#   --max-pieces 32      keep full-board positions (openings included).
#   Previous dataset_sf.pt used max_pieces=20, causing OOD on starting position.
#
#   --n 400000           2x more positions than previous runs (was 200k).
#   With 2.97M available from even sampling, 400k gives better coverage
#   across all game phases and piece counts.
#
# Expected:
#   ~400k positions → output ~1GB .pt file
#   Runtime: ~9h at depth 15 with 16 workers
#   Label std should be lower than dataset_sf.pt (~0.565) — draws and
#   near-equal positions will pull std down, which is expected and correct.

python3 -u /zhome/81/b/206091/Petra-ver2/src/reeval_stockfish.py \
    --dataset   /zhome/81/b/206091/Petra-ver2/data/dataset_feb.pt \
    --out       /zhome/81/b/206091/Petra-ver2/data/dataset_feb_sf.pt \
    --stockfish /zhome/81/b/206091/bin/stockfish \
    --depth     15 \
    --n         400000 \
    --workers   16 \
    --min-decisive 0.0 \
    --max-pieces   32
