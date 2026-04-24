"""
Dataset characterisation: value distribution + piece count (game phase proxy)
for the 2021-06 banded SF-18 datasets.

Usage:
    python3 characterise_datasets.py
"""

import torch
import numpy as np

BLACKHOLE = "/dtu/blackhole/0b/206091"
BANDS = ["low", "mid", "high"]

VALUE_BINS = np.linspace(-1.0, 1.0, 41)   # 40 bins, 0.05 wide
PIECE_BINS = np.arange(2, 33)              # 2..32 pieces on board

def load_split(path, split):
    d = torch.load(path, map_location="cpu", weights_only=False)[split]
    tensors = d["tensors"]   # (N, 14, 8, 8) uint8
    values  = d["values"]    # (N,) float32, tanh-squashed
    return tensors, values

def characterise(band):
    path = f"{BLACKHOLE}/dataset_2021_06_{band}_sf18.pt"
    print(f"\n{'='*60}")
    print(f"  Band: {band.upper()}  ({path})")
    print(f"{'='*60}")

    all_tensors, all_values = [], []
    for split in ("train", "val"):
        t, v = load_split(path, split)
        all_tensors.append(t)
        all_values.append(v)

    tensors = torch.cat(all_tensors)   # (N, 14, 8, 8)
    values  = torch.cat(all_values)    # (N,)
    N = len(values)

    # --- Piece count (planes 0-11 are piece planes) ---
    piece_count = tensors[:, :12].sum(dim=(1, 2, 3)).float()  # (N,)

    # --- Value stats ---
    v = values.numpy()
    pc = piece_count.numpy()

    print(f"\n  Positions: {N:,}")

    print(f"\n  Value distribution (tanh-squashed SF eval):")
    print(f"    Mean:  {v.mean():+.4f}   Std: {v.std():.4f}")
    print(f"    Min:   {v.min():+.4f}   Max: {v.max():+.4f}")
    decisive = (np.abs(v) > 0.5).mean() * 100
    equal    = (np.abs(v) < 0.1).mean() * 100
    mild     = ((np.abs(v) >= 0.1) & (np.abs(v) <= 0.5)).mean() * 100
    print(f"    |v|>0.5 (decisive):  {decisive:5.1f}%")
    print(f"    0.1≤|v|≤0.5 (mild):  {mild:5.1f}%")
    print(f"    |v|<0.1 (equal):     {equal:5.1f}%")

    # Full histogram (8 buckets)
    bucket_edges = [-1.0, -0.7, -0.5, -0.1, 0.1, 0.5, 0.7, 1.0, 1.001]
    bucket_labels = [
        "≤-0.7 (decisive loss)",
        "-0.7–-0.5",
        "-0.5–-0.1 (mild loss)",
        "-0.1– 0.1 (equal)",
        " 0.1– 0.5 (mild win)",
        " 0.5– 0.7",
        "≥ 0.7 (decisive win)",
    ]
    counts, _ = np.histogram(v, bins=bucket_edges)
    print(f"\n  Value bucket breakdown:")
    for label, count in zip(bucket_labels, counts):
        pct = count / N * 100
        bar = "█" * int(pct / 2)
        print(f"    {label}: {pct:5.1f}%  {bar}")

    # --- Piece count / game phase ---
    print(f"\n  Piece count distribution (proxy for game phase):")
    print(f"    Mean: {pc.mean():.1f}   Std: {pc.std():.1f}   Min: {pc.min():.0f}   Max: {pc.max():.0f}")
    opening    = (pc >= 28).mean() * 100
    middlegame = ((pc >= 16) & (pc < 28)).mean() * 100
    late_mid   = ((pc >= 10) & (pc < 16)).mean() * 100
    endgame    = (pc < 10).mean() * 100
    print(f"    ≥28 pieces (opening):       {opening:5.1f}%")
    print(f"    16–27 pieces (middlegame):  {middlegame:5.1f}%")
    print(f"    10–15 pieces (late mid):    {late_mid:5.1f}%")
    print(f"    <10 pieces (endgame):       {endgame:5.1f}%")

    # Piece count histogram in bands of 4
    print(f"\n  Piece count histogram:")
    for lo in range(2, 33, 4):
        hi = min(lo + 3, 32)
        pct = ((pc >= lo) & (pc <= hi)).mean() * 100
        bar = "█" * int(pct / 1)
        print(f"    {lo:2d}–{hi:2d}: {pct:5.1f}%  {bar}")

    # --- Value vs piece count correlation ---
    corr = np.corrcoef(np.abs(v), pc)[0, 1]
    print(f"\n  Correlation |value| vs piece count: {corr:+.4f}")
    print(f"  (negative = fewer pieces → more decisive, as expected for endgames)")

for band in BANDS:
    characterise(band)

print(f"\n{'='*60}")
print("Done.")
