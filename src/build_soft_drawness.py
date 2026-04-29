#!/usr/bin/env python3
"""
build_soft_drawness.py

Compute outcome-smoothed drawness targets for an SF-reevaled dataset.

For each position, drawness_soft_target = fraction of its k nearest geometric
neighbours (in the supplied model's bottleneck) whose game ended in a draw.
This converts drawness from a static/binary label into an empirical
probability — "what fraction of optimally-played games from this geometric
region end in a draw".

Reads:  --dataset (must contain outcome_values)
Writes: dataset["train"]["drawness_soft_targets"]  float32 in [0,1]
        dataset["val"]["drawness_soft_targets"]    float32 in [0,1]

Usage
-----
    python3 src/build_soft_drawness.py \
        --dataset /dtu/blackhole/0b/206091/dataset_drawness_curriculum.pt \
        --model   /zhome/81/b/206091/Petra-ver2/models/phase15_mid_no_endgame/best.pt \
        --out     /dtu/blackhole/0b/206091/dataset_drawness_curriculum_soft.pt \
        --k 50
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PetraNet
from config import device


def extract_geometry(model: PetraNet, tensors: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    model.eval()
    n = len(tensors)
    out = np.empty((n, model.bottleneck_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            t = tensors[i:i + batch_size].float().to(device)
            g = model.geometry(t).cpu().numpy()
            out[i:i + batch_size] = g
    return out


def compute_soft_targets(geo: np.ndarray, outcomes: np.ndarray, k: int) -> np.ndarray:
    """For each row: fraction of its k nearest cosine neighbours where |outcome|<1e-4."""
    from sklearn.neighbors import NearestNeighbors

    is_draw = (np.abs(outcomes) < 1e-4).astype(np.float32)
    print(f"  base draw rate in pool: {is_draw.mean():.3f}")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(geo)
    _, idx = nn.kneighbors(geo, return_distance=True)
    idx = idx[:, 1:]   # drop self
    return is_draw[idx].mean(axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Compute outcome-smoothed drawness targets.")
    ap.add_argument("--dataset", required=True, help="Input curriculum .pt with outcome_values")
    ap.add_argument("--model",   required=True, help="Model whose geometry defines neighbours")
    ap.add_argument("--out",     required=True, help="Output .pt path")
    ap.add_argument("--k",       type=int, default=50, help="Number of neighbours (default: 50)")
    args = ap.parse_args()

    print(f"Loading dataset {args.dataset} ...")
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    for split in ("train", "val"):
        if "outcome_values" not in data[split]:
            sys.exit(f"ERROR: {split} split has no outcome_values — cannot compute soft targets.")

    print(f"Loading model {args.model} ...")
    model = PetraNet().to(device)
    sd = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)

    for split in ("train", "val"):
        d = data[split]
        n = len(d["tensors"])
        print(f"\n[{split}] n={n:,}  extracting geometry ...")
        t0 = time.time()
        geo = extract_geometry(model, d["tensors"])
        print(f"  geometry done in {time.time() - t0:.1f}s — shape {geo.shape}")

        outcomes = d["outcome_values"].numpy()
        print(f"  computing k-NN soft targets (k={args.k}) ...")
        t0 = time.time()
        soft = compute_soft_targets(geo, outcomes, k=args.k)
        print(f"  soft-target NN done in {time.time() - t0:.1f}s")
        print(f"  soft target stats: mean={soft.mean():.3f}  median={np.median(soft):.3f}  "
              f"max={soft.max():.3f}  >0.5: {(soft > 0.5).mean()*100:.1f}%")

        d["drawness_soft_targets"] = torch.from_numpy(soft)

    data.setdefault("meta", {})["soft_drawness"] = {
        "model":  args.model,
        "k":      args.k,
        "metric": "cosine",
    }

    print(f"\nSaving {args.out} ...")
    torch.save(data, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
