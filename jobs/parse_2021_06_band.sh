#!/bin/bash
#BSUB -J parse_2021_06_${BAND:-mid}
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 18:00
#BSUB -o /zhome/81/b/206091/logs/parse_2021_06_%J.out
#BSUB -e /zhome/81/b/206091/logs/parse_2021_06_%J.err

# Banded ELO parse for 2021-06 Lichess dataset.
# Submit three times with BAND=low / mid / high:
#
#   BAND=low  bsub < jobs/parse_2021_06_band.sh   # 1200-1800: decisive positions
#   BAND=mid  bsub < jobs/parse_2021_06_band.sh   # 1800-2300: complex middlegames
#   BAND=high bsub < jobs/parse_2021_06_band.sh   # 2300+:     quality positions
#
# Outputs to $BLACKHOLE/dataset_2021_06_{low,mid,high}.pt
# Next step: reeval each band, then feed exp_c (mid only) and exp_d (low+mid) to train.

BAND="${BAND:-mid}"

case "$BAND" in
    low)  MIN_ELO=1200; MAX_ELO=1800 ;;
    mid)  MIN_ELO=1800; MAX_ELO=2300 ;;
    high) MIN_ELO=2300; MAX_ELO=0    ;;
    *)
        echo "ERROR: BAND must be low, mid, or high (got: ${BAND})"
        exit 1
        ;;
esac

MAX_GAMES=33333

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

PGN_FILE="${BLACKHOLE}/lichess_db_standard_rated_2021-06.pgn.zst"
OUT_FILE="${BLACKHOLE}/dataset_2021_06_${BAND}.pt"

echo "=== Parse 2021-06 band=${BAND} (elo ${MIN_ELO}-${MAX_ELO:-∞}, max_games=${MAX_GAMES}) ==="
echo "PGN: ${PGN_FILE}"
echo "Out: ${OUT_FILE}"
echo

if [ ! -f "${PGN_FILE}" ]; then
    echo "ERROR: PGN not found at ${PGN_FILE}"
    echo "Download with:"
    echo "  wget -P ${BLACKHOLE} https://database.lichess.org/standard/lichess_db_standard_rated_2021-06.pgn.zst"
    exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
pip install zstandard --quiet

MAX_ELO_ARG=""
if [ "${MAX_ELO}" -gt 0 ]; then
    MAX_ELO_ARG="--max-elo ${MAX_ELO}"
fi

python3 -u "${SRC}/data.py" \
    --pgn                "${PGN_FILE}" \
    --out                "${OUT_FILE}" \
    --max-games          ${MAX_GAMES} \
    --min-elo            ${MIN_ELO} \
    ${MAX_ELO_ARG} \
    --positions-per-game 20 \
    --skip-opening       5 \
    --sampling           even \
    --no-strict          \
    --checkpoint-every   10000

echo
echo "Done. Output: ${OUT_FILE}"
echo "Next: submit reeval for this band, then train exp_c (mid only) or exp_d (low+mid)."
