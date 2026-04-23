#!/bin/bash
#BSUB -J merge_2021_06_all
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/merge_2021_06_all_%J.out
#BSUB -e /zhome/81/b/206091/logs/merge_2021_06_all_%J.err

# Merge all three 2021-06 band sf18 datasets into one file for exp_e training.
# Run after all three reeval merge jobs complete.

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"

LOW="${BLACKHOLE}/dataset_2021_06_low_sf18.pt"
MID="${BLACKHOLE}/dataset_2021_06_mid_sf18.pt"
HIGH="${BLACKHOLE}/dataset_2021_06_high_sf18.pt"
OUT="${BLACKHOLE}/dataset_2021_06_all_sf18.pt"

for F in "${LOW}" "${MID}" "${HIGH}"; do
    if [ ! -f "${F}" ]; then
        echo "ERROR: missing file: ${F}"
        exit 1
    fi
done

echo "=== Merging low + mid + high → ${OUT} ==="

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u - <<'PYEOF'
import sys, torch

BLACKHOLE = "/dtu/blackhole/0b/206091"
paths = [
    f"{BLACKHOLE}/dataset_2021_06_low_sf18.pt",
    f"{BLACKHOLE}/dataset_2021_06_mid_sf18.pt",
    f"{BLACKHOLE}/dataset_2021_06_high_sf18.pt",
]
out_path = f"{BLACKHOLE}/dataset_2021_06_all_sf18.pt"

def ensure(d, key, n, dtype, fill=0):
    if key in d:
        return d[key]
    return torch.full((n,) + (() if dtype != torch.uint8 else (512,)), fill, dtype=dtype)

datasets = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]

def merge_split(splits):
    n_each = [len(s["tensors"]) for s in splits]
    result = {
        "tensors":   torch.cat([s["tensors"]   for s in splits]),
        "values":    torch.cat([s["values"]    for s in splits]),
        "move_idxs": torch.cat([s["move_idxs"] for s in splits]),
        "visit_dists": torch.cat([
            s.get("visit_dists", torch.zeros(n, 4096)) for s, n in zip(splits, n_each)
        ]),
        "drawness_mask":      torch.cat([s.get("drawness_mask",     torch.zeros(n, dtype=torch.bool))  for s, n in zip(splits, n_each)]),
        "drawness_targets":   torch.cat([s.get("drawness_targets",  torch.zeros(n))                    for s, n in zip(splits, n_each)]),
        "drawness_available": torch.cat([s.get("drawness_available",torch.zeros(n, dtype=torch.bool))  for s, n in zip(splits, n_each)]),
    }
    if any("legal_masks" in s for s in splits):
        result["legal_masks"] = torch.cat([
            s.get("legal_masks", torch.full((n, 512), 255, dtype=torch.uint8))
            for s, n in zip(splits, n_each)
        ])
    return result

merged = {
    "train": merge_split([d["train"] for d in datasets]),
    "val":   merge_split([d["val"]   for d in datasets]),
}

n_train = len(merged["train"]["tensors"])
n_val   = len(merged["val"]["tensors"])
print(f"Merged: {n_train} train + {n_val} val positions")
torch.save(merged, out_path)
print(f"Saved → {out_path}")
PYEOF

echo
echo "Done. Output: ${OUT}"
echo "Next: bsub < jobs/train_exp_e_gpu.sh"
