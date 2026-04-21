#!/bin/bash
#BSUB -J gen_endgame_raw
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:30
#BSUB -o /zhome/81/b/206091/logs/gen_endgame_raw_%J.out
#BSUB -e /zhome/81/b/206091/logs/gen_endgame_raw_%J.err

# Generate 1M rule-labelled endgame positions (all stages, including drawn).
# Output is used as input to reeval_endgame_sf15.sh (chunked SF re-evaluation).
#
# Stages used:
#   Decisive:  1=KQvK  2=KRvK  3=KPvK  4=KQvKR  5=KRvKP  6=KBvKP  7=KNvKP  8=KPvKP
#   Drawn:     9=KRvKR  10=KNNvK  11=KBvKB
#
# After this job finishes, submit the 4 SF re-evaluation chunks:
#   bsub -env "CHUNK_IDX=0" < jobs/reeval_endgame_sf15.sh
#   bsub -env "CHUNK_IDX=1" < jobs/reeval_endgame_sf15.sh
#   bsub -env "CHUNK_IDX=2" < jobs/reeval_endgame_sf15.sh
#   bsub -env "CHUNK_IDX=3" < jobs/reeval_endgame_sf15.sh

# 250k base positions × 11 stages × ~4 mirror/turn variants ≈ 1M total positions.
# visit_dists are skipped — reeval_stockfish.py only needs FENs; storing 4096
# floats per position would require ~16GB for 1M positions.
N="${N:-250000}"
SEED="${SEED:-42}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"
OUT_FILE="${BLACKHOLE}/endgame_raw.pt"

echo "=== Generating ${N} raw endgame positions (stages 1-11) ==="
echo "Output: ${OUT_FILE}"
echo

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/generate_endgame.py" \
    --positions     ${N} \
    --stages        1 2 3 4 5 6 7 8 9 10 11 \
    --seed          ${SEED} \
    --no-visit-dists \
    --out           "${OUT_FILE}"

echo
echo "Done. Raw dataset: ${OUT_FILE}"
echo
echo "Next — submit 4 SF re-evaluation chunks:"
echo "  bsub -env \"CHUNK_IDX=0\" < jobs/reeval_endgame_sf15.sh"
echo "  bsub -env \"CHUNK_IDX=1\" < jobs/reeval_endgame_sf15.sh"
echo "  bsub -env \"CHUNK_IDX=2\" < jobs/reeval_endgame_sf15.sh"
echo "  bsub -env \"CHUNK_IDX=3\" < jobs/reeval_endgame_sf15.sh"
