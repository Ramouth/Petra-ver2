#!/usr/bin/env python3
"""
build_natural_v4.py

Builds a multi-month natural dataset using OUTCOME VALUES (game results)
as the training target instead of SF eval. AlphaZero-style supervision —
raw +1 / 0 / -1 outcome from STM perspective, no engine teacher signal.

Sources should be from disjoint Lichess months to avoid nested-ELO overlap.
FEN-based deduplication is applied as defense-in-depth.

Usage
-----
    python3 src/build_natural_v4.py \
        --sources \
            /dtu/blackhole/0b/206091/dataset_2021_06_all_sf18.pt \
            /dtu/blackhole/0b/206091/dataset_2023_03_sf18.pt \
            /dtu/blackhole/0b/206091/dataset_2025_01.pt \
        --target-size       5000000 \
        --target-draw-frac  0.075 \
        --out               /dtu/blackhole/0b/206091/dataset_natural_v4.pt
"""

import argparse
import os
import sys

import torch


def _load_split(d, split):
    s = d[split]
    n = len(s["tensors"])
    return s, n


def _index(split, idx):
    out = {}
    for k, v in split.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[idx]
        elif isinstance(v, list):
            out[k] = [v[i] for i in idx.tolist()]
    return out


def _override_values_with_outcome(split: dict) -> dict:
    """Replace `values` with `outcome_values` (STM-relative game outcome).
    Drops positions where outcome_values is missing."""
    if "outcome_values" not in split:
        sys.exit("ERROR: source split missing outcome_values — cannot use outcome supervision")
    out = dict(split)
    out["values"] = split["outcome_values"].float()
    return out


def _dedup_split(s: dict, seen: set) -> tuple[dict, int]:
    if "fens" not in s:
        return s, 0
    fens = s["fens"]
    keep = []
    for i, f in enumerate(fens):
        if f in seen:
            continue
        seen.add(f)
        keep.append(i)
    if len(keep) == len(fens):
        return s, 0
    keep_t = torch.tensor(keep, dtype=torch.long)
    return _index(s, keep_t), len(fens) - len(keep)


def _concat(a, b):
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


def _engineer_split(split, target_size, target_draw_frac, seed):
    g = torch.Generator().manual_seed(seed)
    is_draw = split["values"].float().abs() < 1e-4
    draw_idx     = is_draw.nonzero(as_tuple=True)[0]
    decisive_idx = (~is_draw).nonzero(as_tuple=True)[0]
    n_draw_target     = int(round(target_size * target_draw_frac))
    n_decisive_target = target_size - n_draw_target
    n_draw_take     = min(n_draw_target,     len(draw_idx))
    n_decisive_take = min(n_decisive_target, len(decisive_idx))
    if n_draw_take < n_draw_target:
        print(f"  WARNING: only {len(draw_idx):,} drawn rows available, wanted {n_draw_target:,}")
    if n_decisive_take < n_decisive_target:
        print(f"  WARNING: only {len(decisive_idx):,} decisive rows, wanted {n_decisive_target:,}")
    draw_perm     = draw_idx[torch.randperm(len(draw_idx),     generator=g)[:n_draw_take]]
    decisive_perm = decisive_idx[torch.randperm(len(decisive_idx), generator=g)[:n_decisive_take]]
    sel = torch.cat([draw_perm, decisive_perm])
    sel = sel[torch.randperm(len(sel), generator=g)]
    out = _index(split, sel)
    actual = (out["values"].float().abs() < 1e-4).float().mean().item()
    print(f"  result: n={len(sel):,}  drawn fraction={actual:.3f}")
    return out


def _load_pool(paths, dedup_fens: bool):
    train_pool = None
    val_pool   = None
    seen = set() if dedup_fens else None
    total_dropped = 0
    for p in paths:
        print(f"Loading {p} ...")
        d = torch.load(p, map_location="cpu", weights_only=False)
        for split in ("train", "val"):
            s, n = _load_split(d, split)
            s = _override_values_with_outcome(s)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--target-size",       type=int,   required=True)
    ap.add_argument("--target-draw-frac",  type=float, default=0.075)
    ap.add_argument("--val-frac",          type=float, default=0.05)
    ap.add_argument("--out",               required=True)
    ap.add_argument("--no-dedup",          action="store_true",
                    help="Skip FEN-based deduplication (faster, but trusts source disjointness)")
    ap.add_argument("--seed",              type=int,   default=42)
    args = ap.parse_args()

    print("=" * 60)
    print(f"Pool sources: {len(args.sources)}")
    print(f"Supervision: outcome_values  (game result, STM-relative)")
    train_pool, val_pool = _load_pool(args.sources, dedup_fens=not args.no_dedup)
    print(f"\nPool train: {len(train_pool['tensors']):,}  val: {len(val_pool['tensors']):,}")

    train_draws = (train_pool["values"].float().abs() < 1e-4).sum().item()
    val_draws   = (val_pool["values"].float().abs() < 1e-4).sum().item()
    print(f"Pool drawn (outcome=draw) — train: {train_draws:,} "
          f"({train_draws/len(train_pool['tensors']):.3f})  "
          f"val: {val_draws:,} ({val_draws/len(val_pool['tensors']):.3f})")

    n_val_target = max(1000, int(args.target_size * args.val_frac))
    print(f"\nEngineering train (target n={args.target_size:,}, draw_frac={args.target_draw_frac}):")
    train = _engineer_split(train_pool, args.target_size, args.target_draw_frac, seed=args.seed)
    print(f"\nEngineering val (target n={n_val_target:,}, draw_frac={args.target_draw_frac}):")
    val = _engineer_split(val_pool, n_val_target, args.target_draw_frac, seed=args.seed + 1)

    meta = {
        "n_train": len(train["tensors"]),
        "n_val":   len(val["tensors"]),
        "source":  "natural_v4",
        "supervision": "outcome_values (STM-relative game result)",
        "components": {
            "sources":           args.sources,
            "target_size":       args.target_size,
            "target_draw_frac":  args.target_draw_frac,
            "dedup_fens":        not args.no_dedup,
        },
    }
    print(f"\nSaving {args.out} ...")
    torch.save({"train": train, "val": val, "meta": meta}, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
