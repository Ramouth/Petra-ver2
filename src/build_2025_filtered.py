"""
Filter the 2025 outcome-supervised datasets by piece-count to enrich
the structurally-meaningful signal.

Why
---
characterise_2025 showed that ~85% of "draw" labels in dataset_2025_02.pt
sit on opening / middlegame positions where neither side is structurally
forced toward a draw — players just chose one. Outcome supervision on this
mix can't build draw-aware geometry; the draw cluster is heterogeneous.

Filtering out positions with many pieces (i.e. early-game positions) keeps
the corpus where outcome ≈ position truth: late-middlegame and endgame
positions where the result is more determined by structure than by player
choice.

Default cutoff: keep pc <= 23 (drop true openings ≥28 pieces and early
middlegames 24–27). Roughly halves the corpus per file but draws become
a more structurally coherent class.

Usage
-----
    python3 src/build_2025_filtered.py
    python3 src/build_2025_filtered.py --max-pieces 15  # aggressive
    python3 src/build_2025_filtered.py --max-pieces 27  # gentle
"""

import argparse
import os
import sys

import numpy as np
import torch

BLACKHOLE = "/dtu/blackhole/0b/206091"
DEFAULT_INPUTS = [
    f"{BLACKHOLE}/dataset_2025_02.pt",
    f"{BLACKHOLE}/dataset_2025_04.pt",
]


def piece_count_mask(tensors: torch.Tensor, max_pieces: int) -> torch.Tensor:
    pc = tensors[:, :12].sum(dim=(1, 2, 3)).to(torch.int16)
    return pc <= max_pieces


def filter_split(split: dict, max_pieces: int) -> dict:
    tensors = split["tensors"]
    keep = piece_count_mask(tensors, max_pieces)
    n_in  = len(tensors)
    n_out = int(keep.sum())
    print(f"    keep {n_out:,}/{n_in:,} ({100*n_out/n_in:.2f}%)")

    keep_np = keep.numpy()
    out = {
        "tensors":         tensors[keep],
        "values":          split["values"][keep],
        "outcome_values":  split["outcome_values"][keep] if split.get("outcome_values") is not None
                           else split["values"][keep].clone(),
        "move_idxs":       split["move_idxs"][keep],
        "game_ids":        split["game_ids"][keep] if "game_ids" in split else None,
        "plys":            split["plys"][keep] if "plys" in split else None,
        "fens":            [f for f, k in zip(split["fens"], keep_np) if k],
    }
    return {k: v for k, v in out.items() if v is not None}


def filter_and_collect(path: str, max_pieces: int):
    name = os.path.basename(path)
    print(f"\n{name}")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    d = torch.load(path, map_location="cpu", weights_only=False)
    print("  train:")
    train = filter_split(d["train"], max_pieces)
    print("  val:")
    val   = filter_split(d["val"],   max_pieces)
    return train, val, d.get("meta", {})


def merge(splits_a: dict, splits_b: dict) -> dict:
    out = {}
    for k in splits_a:
        if isinstance(splits_a[k], list):
            out[k] = splits_a[k] + splits_b[k]
        else:
            out[k] = torch.cat([splits_a[k], splits_b[k]])
    return out


def report_outcome_breakdown(name: str, split: dict) -> None:
    v = split["outcome_values"].numpy().astype(np.float32)
    n = len(v)
    n_w = int((v >  0.5).sum())
    n_l = int((v < -0.5).sum())
    n_d = int(((v >= -0.5) & (v <= 0.5)).sum())
    print(f"  {name}: n={n:,}  win={100*n_w/n:.2f}%  "
          f"loss={100*n_l/n:.2f}%  draw={100*n_d/n:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    ap.add_argument("--max-pieces", type=int, default=23,
                    help="Keep positions with piece-count <= this (default: 23)")
    ap.add_argument("--out", default=None,
                    help="Output .pt path. Default: dataset_2025_filtered_pcMM.pt under blackhole")
    args = ap.parse_args()

    out_path = args.out or f"{BLACKHOLE}/dataset_2025_filtered_pc{args.max_pieces:02d}.pt"
    print(f"Filter cutoff: piece-count <= {args.max_pieces}")
    print(f"Output:        {out_path}")

    parts = []
    metas = []
    for p in args.inputs:
        r = filter_and_collect(p, args.max_pieces)
        if r is None:
            sys.exit(1)
        train, val, meta = r
        parts.append((train, val))
        metas.append({"source": os.path.basename(p), **meta})

    merged_train = parts[0][0]
    merged_val   = parts[0][1]
    for t, v in parts[1:]:
        merged_train = merge(merged_train, t)
        merged_val   = merge(merged_val,   v)

    print()
    report_outcome_breakdown("merged train", merged_train)
    report_outcome_breakdown("merged val",   merged_val)

    out = {
        "train": merged_train,
        "val":   merged_val,
        "meta": {
            "source":        "build_2025_filtered.py",
            "max_pieces":    args.max_pieces,
            "n_train":       len(merged_train["tensors"]),
            "n_val":         len(merged_val["tensors"]),
            "inputs":        [os.path.basename(p) for p in args.inputs],
            "input_metas":   metas,
        },
    }
    torch.save(out, out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
