#!/bin/bash
#BSUB -J gpu_smoke
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 0:10
#BSUB -o /zhome/81/b/206091/logs/gpu_smoke_%J.out
#BSUB -e /zhome/81/b/206091/logs/gpu_smoke_%J.err

source /zhome/81/b/206091/petra-env/bin/activate
module load gcc/13.4.0-binutils-2.44

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu 0:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    z = x @ y
    print("matmul ok:", z.shape)
PY
