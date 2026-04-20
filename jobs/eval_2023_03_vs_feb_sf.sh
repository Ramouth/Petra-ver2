#!/bin/bash
#BSUB -J eval_2023_03_vs_feb_sf
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 4:00
#BSUB -o /zhome/81/b/206091/logs/eval_2023_03_vs_feb_sf_%J.out
#BSUB -e /zhome/81/b/206091/logs/eval_2023_03_vs_feb_sf_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

# Head-to-head: lichess_2023_03 vs feb_sf at equal n_sim.
#
# Gate: lichess_2023_03 win rate > 50% confirms ELO improvement despite
# geometry regression (rank 17.1 vs 18.9). If < 50%, the geometry
# regression hurt playing strength and the data pipeline fix is critical.

echo "=== lichess_2023_03 vs feb_sf (head-to-head) ==="
python3 -u /zhome/81/b/206091/Petra-ver2/src/evaluate.py \
    --model          /zhome/81/b/206091/Petra-ver2/models/lichess_2023_03/best.pt \
    --baseline-model /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --step 5 \
    --n-sim 100 \
    --games 200 \
    --workers 16
