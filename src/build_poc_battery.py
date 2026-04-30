#!/usr/bin/env python3
"""
build_poc_battery.py — extract the paired-position drawness gate corpus.

Filters a SF-relabeled dataset to "dead-even middlegame, no transient blunder"
positions, then splits by game outcome:

  drawn class    : outcome_values == 0  (game ended in a draw)
  decisive class : outcome_values != 0  (game ended W/L for some side)

Both classes have |SF eval| < 0.2 — the engine sees them as equal. The model
must distinguish them from the position alone. This is the PoC novelty:
no engine separates "balanced but drifting toward a result" from "structurally
drawn." The gate measures whether Petra does.

Usage
-----
    python3 src/build_poc_battery.py \
        --input  /dtu/blackhole/0b/206091/dataset_2021_06_high_sf18.pt \
        --out    /dtu/blackhole/0b/206091/poc_battery_v0.pt \
        --n-per-class 5000

Design notes
------------
- Uses train+val combined (test models never saw any 2021_06 data — split is
  irrelevant for our purposes).
- Blunder filter: position is excluded if any other position from the same
  game within ±2 plies has |SF eval| differing by > 0.4. Catches transient/
  sharp phases where the engine value is about to swing.
- Per-game cap: at most 3 positions per game_id, to avoid one long drawn
  endgame contributing 50 nearly-identical positions.
- Output is balanced by default (n-per-class for both). If drawn hits its
  natural ceiling first, decisive is downsampled to match.
"""

import argparse
import os
import sys

import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser(description="Build the PoC drawness gate battery.")
    ap.add_argument("--input",          required=True, help="Source SF-relabeled .pt")
    ap.add_argument("--out",            required=True, help="Output .pt path")
    ap.add_argument("--ply-min",        type=int,   default=20)
    ap.add_argument("--ply-max",        type=int,   default=80)
    ap.add_argument("--eval-thresh",    type=float, default=0.2,
                    help="|values| < this → 'dead-even' on the board")
    ap.add_argument("--blunder-window", type=int,   default=2,
                    help="±N plies inspected for blunder filter")
    ap.add_argument("--blunder-delta",  type=float, default=0.4,
                    help="max |Δ value| within window before excluding position")
    ap.add_argument("--per-game-cap",   type=int,   default=1)
    ap.add_argument("--n-per-class",    type=int,   default=2500)
    ap.add_argument("--seed",           type=int,   default=42)
    return ap.parse_args()


def concat_splits(data: dict) -> dict:
    """Concatenate train + val into a single flat dict of arrays / lists."""
    keys_tensor = ["tensors", "values", "outcome_values", "plys", "game_ids",
                   "move_idxs", "legal_masks"]
    out = {}
    for k in keys_tensor:
        if k in data["train"] and k in data["val"]:
            out[k] = torch.cat([data["train"][k], data["val"][k]], dim=0)
    out["fens"] = list(data["train"]["fens"]) + list(data["val"]["fens"])
    return out


def apply_blunder_filter(values: np.ndarray, plys: np.ndarray, game_ids: np.ndarray,
                         window: int, delta: float) -> np.ndarray:
    """
    Mark positions that are 'stable' — i.e. no other indexed position from
    the same game within ±window plies has SF eval differing by more than
    `delta`.

    Returns a bool mask of length N.
    """
    n = len(values)

    # Sort by (game_id, ply) so same-game positions are contiguous and ordered.
    order = np.lexsort((plys, game_ids))
    inv   = np.empty_like(order)
    inv[order] = np.arange(n)

    g_sorted = game_ids[order]
    p_sorted = plys[order]
    v_sorted = values[order]
    stable_sorted = np.ones(n, dtype=bool)

    # Scan within each game, checking the ±window neighborhood by ply.
    i = 0
    while i < n:
        j = i
        while j < n and g_sorted[j] == g_sorted[i]:
            j += 1
        # game spans [i, j)
        gp = p_sorted[i:j]
        gv = v_sorted[i:j]
        for k in range(i, j):
            ply_k = p_sorted[k]
            val_k = v_sorted[k]
            lo = ply_k - window
            hi = ply_k + window
            in_win = (gp >= lo) & (gp <= hi)
            in_win[k - i] = False  # exclude self
            if in_win.any():
                if np.max(np.abs(gv[in_win] - val_k)) > delta:
                    stable_sorted[k] = False
        i = j

    # Map back to original index space.
    stable = np.empty(n, dtype=bool)
    stable[order] = stable_sorted
    return stable


def cap_per_game(idxs: np.ndarray, game_ids: np.ndarray, cap: int,
                 rng: np.random.Generator) -> np.ndarray:
    """
    From `idxs` (positions already passing all filters), keep at most `cap`
    per game_id. Random selection within each game using `rng`.
    """
    chosen = []
    by_game: dict[int, list[int]] = {}
    for i in idxs:
        gid = int(game_ids[i])
        by_game.setdefault(gid, []).append(int(i))
    for gid, lst in by_game.items():
        if len(lst) <= cap:
            chosen.extend(lst)
        else:
            chosen.extend(rng.choice(lst, size=cap, replace=False).tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.input} ...")
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    flat = concat_splits(data)
    n = len(flat["fens"])
    print(f"  Combined train+val: N={n:,}")

    values        = flat["values"].float().numpy()
    outcome_vals  = flat["outcome_values"].float().numpy()
    plys          = flat["plys"].numpy()
    game_ids      = flat["game_ids"].numpy()

    # ── Stage 1: middlegame + dead-even ─────────────────────────────────────
    mid       = (plys >= args.ply_min) & (plys <= args.ply_max)
    dead_even = np.abs(values) < args.eval_thresh
    s1 = mid & dead_even
    print(f"  After ply ∈ [{args.ply_min}, {args.ply_max}] + "
          f"|values|<{args.eval_thresh}: {s1.sum():,}")

    # ── Stage 2: blunder filter ─────────────────────────────────────────────
    print(f"  Applying blunder filter (±{args.blunder_window} plies, "
          f"|Δ| > {args.blunder_delta}) ...")
    # Run blunder filter only on positions that already passed s1 — but we
    # need the full game context (other positions in the same game) to detect
    # blunders. So we run on ALL positions, then mask down.
    stable = apply_blunder_filter(values, plys, game_ids,
                                  window=args.blunder_window,
                                  delta=args.blunder_delta)
    s2 = s1 & stable
    print(f"  After blunder filter: {s2.sum():,} "
          f"(dropped {s1.sum() - s2.sum():,})")

    # ── Stage 3: split by outcome and apply per-game cap ───────────────────
    drawn_idx     = np.where(s2 & (outcome_vals == 0))[0]
    decisive_idx  = np.where(s2 & (outcome_vals != 0))[0]
    print(f"  Pre-cap: drawn={len(drawn_idx):,}  decisive={len(decisive_idx):,}")

    drawn_idx     = cap_per_game(drawn_idx,    game_ids, args.per_game_cap, rng)
    decisive_idx  = cap_per_game(decisive_idx, game_ids, args.per_game_cap, rng)
    print(f"  After per-game cap ({args.per_game_cap}): "
          f"drawn={len(drawn_idx):,}  decisive={len(decisive_idx):,}")

    # ── Stage 4: balance to n_per_class ────────────────────────────────────
    target = min(args.n_per_class, len(drawn_idx), len(decisive_idx))
    print(f"  Balancing to {target:,} per class")
    drawn_keep    = rng.choice(drawn_idx,    size=target, replace=False)
    decisive_keep = rng.choice(decisive_idx, size=target, replace=False)

    keep   = np.concatenate([drawn_keep, decisive_keep])
    labels = np.concatenate([np.zeros(target, dtype=np.int8),    # 0 = drawn
                             np.ones(target,  dtype=np.int8)])   # 1 = decisive
    perm = rng.permutation(len(keep))
    keep   = keep[perm]
    labels = labels[perm]

    # ── Pack output ────────────────────────────────────────────────────────
    out = {
        "tensors":        flat["tensors"][keep].clone(),
        "values":         flat["values"][keep].clone(),
        "outcome_values": flat["outcome_values"][keep].clone(),
        "plys":           flat["plys"][keep].clone(),
        "game_ids":       flat["game_ids"][keep].clone(),
        "labels":         torch.from_numpy(labels),  # 0=drawn, 1=decisive
        "fens":           [flat["fens"][int(i)] for i in keep],
        "meta": {
            "source":          os.path.basename(args.input),
            "ply_range":       [args.ply_min, args.ply_max],
            "eval_thresh":     args.eval_thresh,
            "blunder_window":  args.blunder_window,
            "blunder_delta":   args.blunder_delta,
            "per_game_cap":    args.per_game_cap,
            "n_per_class":     int(target),
            "seed":             args.seed,
            "label_encoding":  "0=drawn (outcome=0), 1=decisive (|outcome|=1)",
        },
    }

    print(f"\nSaving to {args.out} ...")
    torch.save(out, args.out)
    print(f"  N={len(keep):,}  ({target:,} drawn + {target:,} decisive)")
    print(f"  meta: {out['meta']}")


if __name__ == "__main__":
    main()
