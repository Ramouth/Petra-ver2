#!/usr/bin/env python3
"""
build_drawness_curriculum.py

Builds a focused drawness curriculum dataset from multiple SF-reevaled datasets.

Extracts:
  - All drawness positives (drawness_mask=True) across all input datasets
  - A controlled sample of decisive positions (|values| > decisive_threshold)

The decisive positions contribute as implicit negatives via the draw_neg mask
in train.py (|v| > 0.5 → target=0.0) — no explicit drawness_mask needed.

Usage:
    python3 src/build_drawness_curriculum.py \
        --datasets dataset_elo2000_sf18.pt dataset_elo2100_sf18.pt dataset_elo2200_sf18.pt \
        --out dataset_drawness_curriculum.pt \
        --n-decisive 400000 \
        --decisive-threshold 0.5 \
        --val-frac 0.05 \
        --seed 42
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_split(path):
    """Load a dataset and return concatenated train+val fields."""
    data = torch.load(path, map_location="cpu", weights_only=False)

    def cat_optional(key):
        if key in data["train"] and key in data["val"]:
            a, b = data["train"][key], data["val"][key]
            if isinstance(a, torch.Tensor):
                return torch.cat([a, b], dim=0)
        return None

    return {
        "tensors":            torch.cat([data["train"]["tensors"],              data["val"]["tensors"]],              dim=0),
        "values":             torch.cat([data["train"]["values"],               data["val"]["values"]],               dim=0),
        "move_idxs":          torch.cat([data["train"]["move_idxs"],            data["val"]["move_idxs"]],            dim=0),
        "fens":               data["train"]["fens"] + data["val"]["fens"],
        "drawness_mask":      torch.cat([data["train"]["drawness_mask"],        data["val"]["drawness_mask"]],        dim=0),
        "drawness_targets":   torch.cat([data["train"]["drawness_targets"],     data["val"]["drawness_targets"]],     dim=0),
        "drawness_available": torch.cat([data["train"]["drawness_available"],   data["val"]["drawness_available"]],   dim=0),
        "legal_masks":        cat_optional("legal_masks"),
        "outcome_values":     cat_optional("outcome_values"),
        "plys":               cat_optional("plys"),
        "game_ids":           cat_optional("game_ids"),
    }


def main():
    ap = argparse.ArgumentParser(description="Build a focused drawness curriculum dataset.")
    ap.add_argument("--datasets",           nargs="+", required=True,  help="Input SF-reevaled .pt files")
    ap.add_argument("--out",                required=True,             help="Output .pt file")
    ap.add_argument("--n-decisive",         type=int,   default=400_000, help="Number of decisive positions to sample")
    ap.add_argument("--decisive-threshold", type=float, default=0.5,   help="|values| threshold for decisive positions")
    ap.add_argument("--val-frac",           type=float, default=0.05,  help="Fraction of positions for validation")
    ap.add_argument("--seed",               type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── Load all datasets ──────────────────────────────────────────────────
    all_parts = []
    for path in args.datasets:
        print(f"Loading {path} ...")
        part = load_split(path)
        n       = len(part["fens"])
        n_draw  = int(part["drawness_mask"].sum().item())
        n_dec   = int((part["values"].abs() > args.decisive_threshold).sum().item())
        print(f"  {n:,} positions  |  drawness positives: {n_draw:,}  |  decisive: {n_dec:,}")
        all_parts.append(part)

    # ── Concatenate across datasets ────────────────────────────────────────
    def cat_field(key):
        tensors = [p[key] for p in all_parts if p.get(key) is not None]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    tensors          = torch.cat([p["tensors"]            for p in all_parts], dim=0)
    values           = torch.cat([p["values"]             for p in all_parts], dim=0)
    move_idxs        = torch.cat([p["move_idxs"]          for p in all_parts], dim=0)
    fens             = [f for p in all_parts for f in p["fens"]]
    drawness_mask    = torch.cat([p["drawness_mask"]       for p in all_parts], dim=0)
    drawness_targets = torch.cat([p["drawness_targets"]    for p in all_parts], dim=0)
    drawness_avail   = torch.cat([p["drawness_available"]  for p in all_parts], dim=0)
    legal_masks      = cat_field("legal_masks")
    outcome_values   = cat_field("outcome_values")
    plys             = cat_field("plys")

    n_total = len(fens)
    print(f"\nTotal positions across all datasets: {n_total:,}")

    # ── Extract drawness positives ─────────────────────────────────────────
    draw_idxs = drawness_mask.nonzero(as_tuple=True)[0].tolist()
    print(f"Drawness positives: {len(draw_idxs):,}")

    # ── Sample decisive negatives (exclude any drawness positives) ─────────
    decisive_mask = (values.abs() > args.decisive_threshold) & ~drawness_mask
    dec_idxs_all  = decisive_mask.nonzero(as_tuple=True)[0].tolist()
    print(f"Decisive positions available: {len(dec_idxs_all):,}")

    if len(dec_idxs_all) <= args.n_decisive:
        print(f"  Using all {len(dec_idxs_all):,} decisive positions.")
        sampled_dec = dec_idxs_all
    else:
        random.shuffle(dec_idxs_all)
        sampled_dec = dec_idxs_all[:args.n_decisive]
    print(f"Decisive positions sampled: {len(sampled_dec):,}")

    # ── Combine and shuffle ────────────────────────────────────────────────
    combined = draw_idxs + sampled_dec
    random.shuffle(combined)
    n_out = len(combined)

    print(f"\nCurriculum: {n_out:,} total")
    print(f"  drawness positives : {len(draw_idxs):,}  ({100*len(draw_idxs)/n_out:.1f}%)")
    print(f"  decisive negatives : {len(sampled_dec):,}  ({100*len(sampled_dec)/n_out:.1f}%)")

    idx_t   = torch.tensor(combined, dtype=torch.long)
    n_val   = max(1, int(n_out * args.val_frac))
    n_train = n_out - n_val
    train_t = idx_t[:n_train]
    val_t   = idx_t[n_train:]

    def pack(subset):
        d = {
            "tensors":            tensors[subset],
            "values":             values[subset],
            "move_idxs":          move_idxs[subset],
            "fens":               [fens[i] for i in subset.tolist()],
            "drawness_mask":      drawness_mask[subset],
            "drawness_targets":   drawness_targets[subset],
            "drawness_available": drawness_avail[subset],
        }
        if legal_masks     is not None: d["legal_masks"]    = legal_masks[subset]
        if outcome_values  is not None: d["outcome_values"] = outcome_values[subset]
        if plys            is not None: d["plys"]           = plys[subset]
        return d

    meta = {
        "n_train":            n_train,
        "n_val":              n_val,
        "label_type":         "stockfish_tanh_cp400",
        "has_visit_dists":    False,
        "n_drawness":         len(draw_idxs),
        "n_decisive":         len(sampled_dec),
        "decisive_threshold": args.decisive_threshold,
        "source_datasets":    args.datasets,
    }

    torch.save({"train": pack(train_t), "val": pack(val_t), "meta": meta}, args.out)
    print(f"\nSaved → {args.out}")
    print(f"  train: {n_train:,}  val: {n_val:,}")


if __name__ == "__main__":
    main()
