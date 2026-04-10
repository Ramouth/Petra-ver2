#!/bin/bash
#BSUB -J geometry_test
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 0:15
#BSUB -o /zhome/81/b/206091/logs/geometry_test_%J.out
#BSUB -e /zhome/81/b/206091/logs/geometry_test_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

python3 -u /zhome/81/b/206091/Petra-ver2/src/test_geometry.py \
    --model /zhome/81/b/206091/Petra-ver2/models/best.pt
