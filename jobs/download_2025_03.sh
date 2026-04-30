#!/bin/bash
set -euo pipefail
#BSUB -J download_2025_03
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/download_2025_03_%J.out
#BSUB -e /zhome/81/b/206091/logs/download_2025_03_%J.err

# Re-download the truncated Lichess 2025-03 PGN.
# Existing blackhole file is 1.7G — full month is ~28G. Download appears
# to have been interrupted; remove and fetch fresh.
#
#   bsub < jobs/download_2025_03.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
URL="https://database.lichess.org/standard/lichess_db_standard_rated_2025-03.pgn.zst"
OUT="${BLACKHOLE}/lichess_db_standard_rated_2025-03.pgn.zst"

echo "=== Re-downloading Lichess 2025-03 PGN ==="
echo "URL: ${URL}"
echo "Out: ${OUT}"
echo

if [ -f "${OUT}" ]; then
    SIZE=$(du -h "${OUT}" | cut -f1)
    echo "Removing existing (likely truncated) file: ${OUT} (${SIZE})"
    rm "${OUT}"
fi

wget --progress=dot:giga -O "${OUT}" "${URL}"

echo
echo "Download complete:"
ls -lh "${OUT}"
