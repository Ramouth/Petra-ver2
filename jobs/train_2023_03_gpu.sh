#!/bin/bash
#BSUB -J train_2023_03
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_2023_03_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_2023_03_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

# Training on Lichess 2023-03, 2000+ ELO, SF depth-18 labels.
#
# Key changes vs feb_sf:
#   - Initialised from feb_sf/best.pt — fine-tunes the existing geometry
#   - 2000+ ELO filter (vs 2500) — broader value distribution, more decisive
#     games, better endgame calibration
#   - draw label 0.0 (was -0.1 in feb_sf game-outcome data)
#   - epochs=50: let geo-patience=3 control stopping, not a hard ceiling
#   - anchor-dataset=dataset_feb_sf.pt (15%): prevents forgetting draw concept
#     (KR vs KR drifted to -0.581 without anchor — real games lack drawn endgames)
#   - lr=5e-4 (was 1e-3): fine-tuning LR; gradient norms grew 0.95→1.90 without it
#
# Geometry gate (run probe_geometry_2023_03.sh after training):
#   - Effective rank > 20 (target: >30)
#   - win·draw cosine (strict) < 0.1676 (feb_sf baseline)
#   - loss·draw cosine < 0.4550 (lichess_2023_03 baseline — draws must separate)

BLACKHOLE="/dtu/blackhole/0b/206091"

python3 -u /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset        ${BLACKHOLE}/dataset_2023_03_sf18.pt \
    --init-model     /zhome/81/b/206091/Petra-ver2/models/feb_sf/best.pt \
    --anchor-dataset /zhome/81/b/206091/Petra-ver2/data/dataset_feb_sf.pt \
    --anchor-frac    0.15 \
    --rank-reg       0.1 \
    --num-workers    0 \
    --weight-decay   5e-4 \
    --lr             5e-4 \
    --epochs         50 \
    --out            /zhome/81/b/206091/Petra-ver2/models/lichess_2023_03
