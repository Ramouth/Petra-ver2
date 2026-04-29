#!/bin/bash
# Show progress of all reeval_elo_ablation log files.
# Run from gbarlogin: bash jobs/status_reeval_elo.sh

for f in ~/logs/reeval_elo_ablation_*.out; do
    echo "=== $(basename $f) ==="
    tail -4 "$f"
    echo
done
