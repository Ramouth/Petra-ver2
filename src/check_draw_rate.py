#!/usr/bin/env python3
"""Quick check: game-outcome draw rate in a parsed dataset."""
import sys
import torch

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/dtu/blackhole/0b/206091/dataset_dec_ablation_raw.pt"

d = torch.load(path, map_location="cpu")
v = d["outcome_values"]

draw = (v == 0).float().mean().item()
dec  = (v.abs() > 0.5).float().mean().item()

print(f"Dataset:                       {path}")
print(f"Total positions:               {len(v):,}")
print(f"Positions from drawn games:    {draw*100:.1f}%")
print(f"Positions from decisive games: {dec*100:.1f}%")
