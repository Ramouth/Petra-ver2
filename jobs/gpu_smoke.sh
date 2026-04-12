#!/bin/bash
#BSUB -J gpu_smoke
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=2GB] span[hosts=1]"
#BSUB -W 0:10
#BSUB -o /zhome/81/b/206091/logs/gpu_smoke_%J.out
#BSUB -e /zhome/81/b/206091/logs/gpu_smoke_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44
module load cuda/11.8

echo "--- env loaded ---"
nvidia-smi

python3 /zhome/81/b/206091/Petra-ver2/src/gpu_smoke.py
