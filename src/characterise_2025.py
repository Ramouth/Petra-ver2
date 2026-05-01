"""
Characterise the 2025 outcome-supervised datasets used by train_big_2025.sh.

Unlike characterise_datasets.py (which expects continuous tanh-squashed SF
values), this expects discrete game-outcome labels — typically {+1, -1, 0}
or the data.py convention {+1, -1, -0.1}. It groups positions by outcome
class and reports phase distribution within each class.

The drawness PoC question: are draw-outcome positions concentrated in
endgames, or scattered across all phases? If draws are mostly opening/
middlegame agreed-draws, outcome supervision can't build draw-aware
geometry — the "draw" cluster is structurally heterogeneous.

Usage
-----
    python3 src/characterise_2025.py
    python3 src/characterise_2025.py --paths /dtu/blackhole/0b/206091/dataset_2025_02.pt /dtu/blackhole/0b/206091/dataset_2025_04.pt
"""

import argparse
import gc
import os
import sys

import numpy as np
import torch

DEFAULT_PATHS = [
    "/dtu/blackhole/0b/206091/dataset_2025_02.pt",
    "/dtu/blackhole/0b/206091/dataset_2025_04.pt",
]


def classify(v: np.ndarray) -> np.ndarray:
    """Return 'win'/'loss'/'draw' per position. Tolerant to {+1,-1,0} or {+1,-1,-0.1}."""
    out = np.full(v.shape, "?", dtype=object)
    out[v > 0.5]                          = "win"
    out[v < -0.5]                         = "loss"
    out[(v >= -0.5) & (v <= 0.5)]         = "draw"
    return out


def load_split(path: str, split: str):
    d = torch.load(path, map_location="cpu", weights_only=False)[split]
    tensors = d["tensors"]
    # Prefer outcome_values when present — that's what big_2025 trains on.
    if "outcome_values" in d and d["outcome_values"] is not None:
        values = d["outcome_values"]
        src = "outcome_values"
    else:
        values = d["values"]
        src = "values"
    return tensors, values, src


def characterise(path: str) -> None:
    name = os.path.basename(path)
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return

    # Stream split-by-split to avoid holding both train and val tensors
    # at full scale alongside torch.load's other dict entries.
    pieces_chunks = []
    values_chunks = []
    src = None
    for split in ("train", "val"):
        try:
            t, v, s = load_split(path, split)
        except KeyError as e:
            print(f"  ERROR reading split={split}: {e}")
            return
        src = s
        pieces_chunks.append(t[:, :12].sum(dim=(1, 2, 3)).to(torch.int16).numpy())
        values_chunks.append(v.numpy().astype(np.float32))
        del t, v
        gc.collect()

    pieces = np.concatenate(pieces_chunks).astype(np.float32)
    values = np.concatenate(values_chunks)
    del pieces_chunks, values_chunks
    gc.collect()
    N = len(values)

    print(f"\n  Positions: {N:,}   (label source: {src})")
    print(f"  Raw value stats: min={values.min():+.4f}  max={values.max():+.4f}  "
          f"mean={values.mean():+.4f}  std={values.std():.4f}")
    uniq, counts = np.unique(np.round(values, 4), return_counts=True)
    print(f"  Distinct label values: {len(uniq)}")
    if len(uniq) <= 8:
        for u, c in zip(uniq, counts):
            print(f"    {u:+.4f}: {c:>10,}  ({100*c/N:5.2f}%)")

    cls = classify(values)
    n_win  = (cls == "win").sum()
    n_loss = (cls == "loss").sum()
    n_draw = (cls == "draw").sum()
    print(f"\n  Outcome classes:")
    print(f"    win:   {n_win:>10,}  ({100*n_win/N:5.2f}%)")
    print(f"    loss:  {n_loss:>10,}  ({100*n_loss/N:5.2f}%)")
    print(f"    draw:  {n_draw:>10,}  ({100*n_draw/N:5.2f}%)")

    print(f"\n  Piece-count overall: mean={pieces.mean():.1f}  "
          f"min={pieces.min():.0f}  max={pieces.max():.0f}")

    def phase_breakdown(label: str, mask: np.ndarray) -> None:
        if mask.sum() == 0:
            return
        p = pieces[mask]
        n = len(p)
        opening    = (p >= 28).mean() * 100
        middlegame = ((p >= 16) & (p < 28)).mean() * 100
        late_mid   = ((p >= 10) & (p < 16)).mean() * 100
        endgame    = (p < 10).mean() * 100
        print(f"\n  Phase distribution within '{label}' (n={n:,}):")
        print(f"    >=28 pieces (opening):       {opening:5.2f}%")
        print(f"    16-27 pieces (middlegame):   {middlegame:5.2f}%")
        print(f"    10-15 pieces (late mid):     {late_mid:5.2f}%")
        print(f"    <10 pieces (endgame):        {endgame:5.2f}%")

    phase_breakdown("draw", cls == "draw")
    phase_breakdown("win",  cls == "win")
    phase_breakdown("loss", cls == "loss")

    # PoC-relevant question: where are the draws concentrated?
    if n_draw > 0:
        draw_pieces = pieces[cls == "draw"]
        all_pieces  = pieces
        print(f"\n  Draw vs overall piece-count:")
        print(f"    draw mean pieces:    {draw_pieces.mean():.1f}")
        print(f"    overall mean pieces: {all_pieces.mean():.1f}")
        print(f"    delta:               {draw_pieces.mean() - all_pieces.mean():+.2f}")
        print(f"    (negative = draws skew toward endgame; positive = toward opening/midgame)")

    del pieces, values, cls
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=DEFAULT_PATHS)
    args = ap.parse_args()

    for p in args.paths:
        characterise(p)

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
