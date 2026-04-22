#!/bin/bash
#BSUB -J reeval_low_elo
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB] select[hname!='n-62-21-89']"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/reeval_low_elo_%J.out
#BSUB -e /zhome/81/b/206091/logs/reeval_low_elo_%J.err

# SF depth-18 re-evaluation of the low-ELO (1600-1850) dataset.
#
# Dataset: ~100k games × 20 positions = ~2M positions.
# We sample 500k for SF reeval (~5 positions/game, enough for game-level
# drawness check). Throughput: ~20k pos/h with 16 workers → ~25h → 24h wall.
#
# Game-level drawness: a game is marked only if ALL sampled positions had
# |SF eval| < 0.11. This excludes blunder-to-draw games.
#
# After this job, train with train_low_elo_gpu.sh.
#
# Submit:
#   bsub < jobs/reeval_low_elo.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

IN_FILE="${BLACKHOLE}/dataset_2025_01_low_elo.pt"
OUT_FILE="${BLACKHOLE}/dataset_2025_01_low_elo_sf18.pt"

echo "=== SF depth-18 re-evaluation: 2025-01 low-ELO (1600-1850) ==="
echo "In:  ${IN_FILE}"
echo "Out: ${OUT_FILE}"
echo

if [ ! -f "${IN_FILE}" ]; then
    echo "ERROR: dataset not found: ${IN_FILE}"
    echo "Run parse_low_elo_2025_01.sh first"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44

python3 -u "${SRC}/reeval_stockfish.py" \
    --dataset                      "${IN_FILE}" \
    --stockfish                    "${HOME_DIR}/bin/stockfish" \
    --depth                        18 \
    --n                            500000 \
    --workers                      16 \
    --out                          "${OUT_FILE}" \
    --derive-drawness-from-outcome \
    --drawness-game-level \
    --drawness-sf-threshold        0.11 \
    --drawness-target              0.8

echo
echo "Done. Dataset: ${OUT_FILE}"
echo "Next: bsub < jobs/train_low_elo_gpu.sh"
