#!/usr/bin/env python3
"""
build_natural_v2.py

Builds the natural v2 dataset: all 2100+ ELO positions + drawn games from 2200+.

No sustained-equality filtering, no synthetic endgames, no scaffolding labels.
Geometry learns drawness from the natural value distribution at high ELO,
boosted by extra drawn games from the strongest band.

Usage
-----
    python3 src/build_natural_v2.py \
        --base-dataset /dtu/blackhole/0b/206091/dataset_elo2100_sf18.pt \
        --draw-dataset /dtu/blackhole/0b/206091/dataset_elo2200_sf18.pt \
        --out          /dtu/blackhole/0b/206091/dataset_natural_v2.pt
"""

import argparse
import os
import sys

import torch


REQUIRED_FIELDS = ("tensors", "values", "move_idxs")
OPTIONAL_FIELDS = ("legal_masks", "outcome_values", "plys", "game_ids",
                   "drawness_mask", "drawness_targets", "drawness_available", "fens")


def _filter_drawn(split: dict) -> dict:
    """Keep only rows from games whose outcome was a draw."""
    if "outcome_values" not in split:
        sys.exit("ERROR: draw-dataset split has no outcome_values — cannot filter drawn games.")
    is_draw = split["outcome_values"].float().abs() < 1e-4
    keep = is_draw.nonzero(as_tuple=True)[0]
    print(f"    drawn rows: {len(keep):,} / {len(split['tensors']):,}")
    out = {}
    for k, v in split.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[keep]
        elif isinstance(v, list):
            out[k] = [v[i] for i in keep.tolist()]
        else:
            out[k] = v
    return out


def _concat_split(a: dict, b: dict) -> dict:
    """Concatenate two splits row-wise. Drops fields not present in both."""
    common = set(a.keys()) & set(b.keys())
    out = {}
    for k in common:
        va, vb = a[k], b[k]
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            out[k] = torch.cat([va, vb], dim=0)
        elif isinstance(va, list) and isinstance(vb, list):
            out[k] = va + vb
    return out


def _shuffle(split: dict, seed: int = 42) -> dict:
    g = torch.Generator().manual_seed(seed)
    n = len(split["tensors"])
    perm = torch.randperm(n, generator=g)
    out = {}
    for k, v in split.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[perm]
        elif isinstance(v, list):
            out[k] = [v[i] for i in perm.tolist()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dataset",  required=True, help="2100+ ELO SF-reeval'd dataset (full)")
    ap.add_argument("--draw-dataset",  required=True, help="2200+ ELO SF-reeval'd dataset (drawn-only filter applied)")
    ap.add_argument("--out",           required=True)
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    print(f"Loading base: {args.base_dataset}")
    base = torch.load(args.base_dataset, map_location="cpu", weights_only=False)
    print(f"  train: {len(base['train']['tensors']):,}  val: {len(base['val']['tensors']):,}")

    print(f"\nLoading draw source: {args.draw_dataset}")
    draw = torch.load(args.draw_dataset, map_location="cpu", weights_only=False)
    print(f"  train: {len(draw['train']['tensors']):,}  val: {len(draw['val']['tensors']):,}")

    print("\nFiltering draw-source to outcome=draw rows ...")
    print("  train:")
    draw_train = _filter_drawn(draw["train"])
    print("  val:")
    draw_val = _filter_drawn(draw["val"])

    print("\nConcatenating ...")
    train = _shuffle(_concat_split(base["train"], draw_train), seed=args.seed)
    val   = _shuffle(_concat_split(base["val"],   draw_val),   seed=args.seed + 1)
    print(f"  total train: {len(train['tensors']):,}")
    print(f"  total val:   {len(val['tensors']):,}")

    n_draw_train = int((train["outcome_values"].float().abs() < 1e-4).sum().item()) if "outcome_values" in train else 0
    n_total = len(train["tensors"])
    print(f"  drawn fraction in train: {n_draw_train/n_total:.3f}")

    meta = {
        "n_train": len(train["tensors"]),
        "n_val":   len(val["tensors"]),
        "source":  "natural_v2",
        "components": {
            "base_dataset":  args.base_dataset,
            "draw_dataset":  args.draw_dataset,
            "draw_filter":   "outcome_values == draw",
        },
    }

    print(f"\nSaving {args.out} ...")
    torch.save({"train": train, "val": val, "meta": meta}, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
