#!/bin/bash
#BSUB -J merge_phase15_special
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 1:00
#BSUB -o /zhome/81/b/206091/logs/merge_phase15_special_%J.out
#BSUB -e /zhome/81/b/206091/logs/merge_phase15_special_%J.err

# Produce 4 merged ablation files for Phase 1.5 comparison training.
# Each file is a different combination of special datasets so that
# train.py --extra-dataset can swap them without re-merging.
#
# Source dataset position counts (train / val):
#   endgame_phase15_50k.pt        172,230 / 19,136  (191,366 total)
#   dataset_near_mate.pt           60,756 /  3,197   (63,953 total)
#   dataset_near_mate_losing.pt    30,885 /  1,625   (32,510 total)
#   dataset_material_imbalance.pt 190,000 / 10,000  (200,000 total)
#
# Ablation files (train positions):
#   special_full:        453,871  (all four)
#   special_no_endgame:  281,641  (near_mate + near_mate_losing + material)
#   special_no_nearmate: 362,230  (endgame + material)
#   special_no_material: 263,871  (endgame + near_mate + near_mate_losing)
#
# Mid-band available (~982k positions from partial reeval):
#   reeval_phase15_mid_d18_part0  ~160k
#   reeval_phase15_mid_d18_part1  ~662k  (complete)
#   reeval_phase15_mid_d18_part2  ~160k
#
# Training conditions (6 total):
#   1. mid only              --dataset mid_sf18
#   2. special only (no mid) --dataset special_full
#   3. mid + full special    --dataset mid_sf18  --extra-dataset special_full
#   4. mid + no endgame      --dataset mid_sf18  --extra-dataset special_no_endgame
#   5. mid + no near-mate    --dataset mid_sf18  --extra-dataset special_no_nearmate
#   6. mid + no material     --dataset mid_sf18  --extra-dataset special_no_material
#
# Submit after gen_endgame_phase15.sh (50k) completes.
#
#   bsub < jobs/merge_phase15_special.sh

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"

ENDGAME="${HOME_DIR}/Petra-ver2/data/endgame_phase15_50k.pt"
NEAR_MATE="${BLACKHOLE}/dataset_near_mate.pt"
NEAR_MATE_LOSING="${BLACKHOLE}/dataset_near_mate_losing.pt"
MATERIAL="${BLACKHOLE}/dataset_material_imbalance.pt"

for F in "${ENDGAME}" "${NEAR_MATE}" "${NEAR_MATE_LOSING}" "${MATERIAL}"; do
    if [ ! -f "${F}" ]; then
        echo "ERROR: missing file: ${F}"
        exit 1
    fi
done

echo "=== Generating Phase 1.5 ablation datasets ==="
echo

source "${HOME_DIR}/petra-env/bin/activate"

python3 -u - <<'PYEOF'
import torch

BLACKHOLE = "/dtu/blackhole/0b/206091"
HOME_DIR  = "/zhome/81/b/206091"

endgame      = f"{HOME_DIR}/Petra-ver2/data/endgame_phase15_50k.pt"
near_mate    = f"{BLACKHOLE}/dataset_near_mate.pt"
near_mate_l  = f"{BLACKHOLE}/dataset_near_mate_losing.pt"
material     = f"{BLACKHOLE}/dataset_material_imbalance.pt"

ablations = {
    "special_full":         [endgame, near_mate, near_mate_l, material],
    "special_no_endgame":   [near_mate, near_mate_l, material],
    "special_no_nearmate":  [endgame, material],
    "special_no_material":  [endgame, near_mate, near_mate_l],
}

def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def merge_split(splits):
    n_each = [len(s["tensors"]) for s in splits]
    result = {
        "tensors":   torch.cat([s["tensors"]   for s in splits]),
        "values":    torch.cat([s["values"]    for s in splits]),
        "move_idxs": torch.cat([s["move_idxs"] for s in splits]),
        "drawness_mask":      torch.cat([s.get("drawness_mask",      torch.zeros(n, dtype=torch.bool)) for s, n in zip(splits, n_each)]),
        "drawness_targets":   torch.cat([s.get("drawness_targets",   torch.zeros(n))                   for s, n in zip(splits, n_each)]),
        "drawness_available": torch.cat([s.get("drawness_available", torch.zeros(n, dtype=torch.bool)) for s, n in zip(splits, n_each)]),
    }
    if any("visit_dists" in s for s in splits):
        result["visit_dists"] = torch.cat([
            s.get("visit_dists", torch.zeros(n, 4096)) for s, n in zip(splits, n_each)
        ])
    if any("legal_masks" in s for s in splits):
        result["legal_masks"] = torch.cat([
            s.get("legal_masks", torch.full((n, 512), 255, dtype=torch.uint8))
            for s, n in zip(splits, n_each)
        ])
    return result

cache = {}
def get(path):
    if path not in cache:
        cache[path] = load(path)
    return cache[path]

for name, paths in ablations.items():
    out = f"{BLACKHOLE}/dataset_{name}.pt"
    print(f"\n--- {name} ---")
    datasets = [get(p) for p in paths]
    for p, d in zip(paths, datasets):
        n = len(d["train"]["tensors"])
        print(f"  {p.split('/')[-1]}: train={n:,}")
    merged = {
        "train": merge_split([d["train"] for d in datasets]),
        "val":   merge_split([d["val"]   for d in datasets]),
    }
    n_train = len(merged["train"]["tensors"])
    n_val   = len(merged["val"]["tensors"])
    v = merged["train"]["values"]
    print(f"  → {n_train:,} train + {n_val:,} val  "
          f"(decisive={( v.abs()>=0.7).float().mean()*100:.1f}%  "
          f"mean={v.mean():.3f})")
    torch.save(merged, out)
    print(f"  Saved → {out}")

print("\nDone.")
PYEOF

echo
echo "Outputs in ${BLACKHOLE}:"
ls -lh "${BLACKHOLE}"/dataset_special_*.pt 2>/dev/null || ls -lh "${BLACKHOLE}"/dataset_phase15_special*.pt 2>/dev/null
