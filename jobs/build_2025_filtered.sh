#!/bin/bash
#BSUB -J build_2025_filtered
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 2:00
#BSUB -o /zhome/81/b/206091/logs/build_2025_filtered_%J.out
#BSUB -e /zhome/81/b/206091/logs/build_2025_filtered_%J.err

# Filter dataset_2025_02 + dataset_2025_04 by piece-count to enrich the
# structurally-meaningful signal for outcome supervision. See
# project_outcome_draw_composition memory + characterise_2025 output.
#
# Default cutoff: pc <= 23 (keeps late-mid + endgame; drops true openings
# and early middlegame agreed-draws). Output:
#   /dtu/blackhole/0b/206091/dataset_2025_filtered_pc23.pt
#
# Submit:
#   bsub < jobs/build_2025_filtered.sh

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/build_2025_filtered.py" --max-pieces 23
