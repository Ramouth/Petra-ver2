#!/bin/bash
#BSUB -J characterise_2021_06
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/characterise_2021_06_%J.out
#BSUB -e /zhome/81/b/206091/logs/characterise_2021_06_%J.err

HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u "${SRC}/characterise_datasets.py"
