#!/usr/bin/env python3
"""
build_natural_v3.py

Builds a giant natural-style training dataset by pooling multiple SF-reeval'd
sources, then sampling to a target size while controlling the drawn fraction.

No drawness labels, no curation — just position + value + policy + outcome,
with the draw fraction engineered by oversampling drawn positions and
undersampling decisive ones.

Usage
-----
    python3 src/build_natural_v3.py \
        --sources \
            /dtu/blackhole/0b/206091/dataset_elo2100_sf18.pt \
            /dtu/blackhole/0b/206091/dataset_elo2200_sf18.pt \
            /dtu/blackhole/0b/206091/dataset_2021_06_all_sf18.pt \
            /dtu/blackhole/0b/206091/dataset_2021_06_high_sf18.pt \
            /dtu/blackhole/0b/206091/dataset_elo2000_sf18.pt \
        --target-size  4500000 \
        --target-draw-frac 0.075 \
        --out /dtu/blackhole/0b/206091/dataset_natural_v3.pt
"""

import argparse
import os
import sys

import torch


def _load_split(d, split):
    s = d[split]
    n = len(s["tensors"])
    return s, n


def _load_pool(paths, dedup_fens: bool = True):
    """Load all sources and concatenate. With dedup_fens, skip rows whose FEN was
    already seen (across any prior source or split). Defends against nested-ELO
    sources or cross-month coincidences inflating common positions."""
    train_pool = None
    val_pool   = None
    seen = set() if dedup_fens else None
    total_dropped = 0

    for p in paths:
        print(f"Loading {p} ...")
        d = torch.load(p, map_location="cpu", weights_only=False)
        for split in ("train", "val"):
            s, n = _load_split(d, split)
            if dedup_fens:
                s, dropped = _dedup_split(s, seen)
                total_dropped += dropped
                print(f"  {split}: n={n:,}  kept={len(s['tensors']):,}  dropped={dropped:,}")
            else:
                print(f"  {split}: n={n:,}")
            if split == "train":
                train_pool = _concat(train_pool, s)
            else:
                val_pool = _concat(val_pool, s)

    if dedup_fens:
        print(f"\nDedup total dropped: {total_dropped:,}")
    return train_pool, val_pool


def _dedup_split(s: dict, seen: set) -> tuple[dict, int]:
    """Keep only rows whose FEN is not yet in `seen`. Mutates `seen`."""
    if "fens" not in s:
        return s, 0
    fens = s["fens"]
    keep_idx = []
    for i, f in enumerate(fens):
        if f in seen:
            continue
        seen.add(f)
        keep_idx.append(i)
    if len(keep_idx) == len(fens):
        return s, 0
    keep = torch.tensor(keep_idx, dtype=torch.long)
    return _index(s, keep), len(fens) - len(keep_idx)


def _concat(a, b):
    """Concatenate row-wise; intersect keys so we don't introduce ragged fields."""
    if a is None:
        return {k: v for k, v in b.items() if isinstance(v, (torch.Tensor, list))}
    common = set(a.keys()) & set(b.keys())
    out = {}
    for k in common:
        va, vb = a[k], b[k]
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            out[k] = torch.cat([va, vb], dim=0)
        elif isinstance(va, list) and isinstance(vb, list):
            out[k] = va + vb
    return out


def _index(split, idx):
    out = {}
    for k, v in split.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[idx]
        elif isinstance(v, list):
            out[k] = [v[i] for i in idx.tolist()]
    return out


def _engineer_split(split, target_size, target_draw_frac, seed):
    """Subsample drawn and decisive separately to hit target_size at target_draw_frac."""
    g = torch.Generator().manual_seed(seed)

    if "outcome_values" not in split:
        sys.exit("ERROR: source split missing outcome_values — cannot engineer draw fraction")

    is_draw = split["outcome_values"].float().abs() < 1e-4
    draw_idx     = is_draw.nonzero(as_tuple=True)[0]
    decisive_idx = (~is_draw).nonzero(as_tuple=True)[0]

    n_draw_target     = int(round(target_size * target_draw_frac))
    n_decisive_target = target_size - n_draw_target

    n_draw_avail     = len(draw_idx)
    n_decisive_avail = len(decisive_idx)

    n_draw_take     = min(n_draw_target,     n_draw_avail)
    n_decisive_take = min(n_decisive_target, n_decisive_avail)

    if n_draw_take < n_draw_target:
        print(f"  WARNING: only {n_draw_avail:,} drawn rows available, wanted {n_draw_target:,}")
    if n_decisive_take < n_decisive_target:
        print(f"  WARNING: only {n_decisive_avail:,} decisive rows available, wanted {n_decisive_target:,}")

    draw_perm     = draw_idx[torch.randperm(n_draw_avail,     generator=g)[:n_draw_take]]
    decisive_perm = decisive_idx[torch.randperm(n_decisive_avail, generator=g)[:n_decisive_take]]
    sel = torch.cat([draw_perm, decisive_perm])
    sel = sel[torch.randperm(len(sel), generator=g)]   # shuffle final

    out = _index(split, sel)
    actual_draw_frac = (out["outcome_values"].float().abs() < 1e-4).float().mean().item()
    print(f"  result: n={len(sel):,}  drawn={n_draw_take:,} ({actual_draw_frac:.3f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="SF-reevaled datasets to pool (must have outcome_values)")
    ap.add_argument("--target-size",       type=int,   required=True,
                    help="Target train rows. Val is sampled at val-frac of this.")
    ap.add_argument("--target-draw-frac",  type=float, default=0.075,
                    help="Target drawn fraction (default: 0.075)")
    ap.add_argument("--val-frac",          type=float, default=0.05,
                    help="Val fraction of target_size (default: 0.05)")
    ap.add_argument("--out",               required=True)
    ap.add_argument("--seed",              type=int,   default=42)
    args = ap.parse_args()

    print("=" * 60)
    print(f"Pool sources: {len(args.sources)}")
    train_pool, val_pool = _load_pool(args.sources)
    print(f"\nPool train: {len(train_pool['tensors']):,}  val: {len(val_pool['tensors']):,}")

    train_draws = (train_pool["outcome_values"].float().abs() < 1e-4).sum().item()
    val_draws   = (val_pool["outcome_values"].float().abs()   < 1e-4).sum().item()
    print(f"Pool drawn — train: {train_draws:,} ({train_draws/len(train_pool['tensors']):.3f})  "
          f"val: {val_draws:,} ({val_draws/len(val_pool['tensors']):.3f})")

    n_val_target = max(1000, int(args.target_size * args.val_frac))
    print(f"\nEngineering train (target n={args.target_size:,}, draw_frac={args.target_draw_frac}):")
    train = _engineer_split(train_pool, args.target_size, args.target_draw_frac, seed=args.seed)
    print(f"\nEngineering val (target n={n_val_target:,}, draw_frac={args.target_draw_frac}):")
    val = _engineer_split(val_pool, n_val_target, args.target_draw_frac, seed=args.seed + 1)

    meta = {
        "n_train": len(train["tensors"]),
        "n_val":   len(val["tensors"]),
        "source":  "natural_v3",
        "components": {
            "sources":           args.sources,
            "target_size":       args.target_size,
            "target_draw_frac":  args.target_draw_frac,
            "val_frac":          args.val_frac,
        },
    }

    print(f"\nSaving {args.out} ...")
    torch.save({"train": train, "val": val, "meta": meta}, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
