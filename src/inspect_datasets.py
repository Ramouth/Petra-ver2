#!/usr/bin/env python3
"""
inspect_datasets.py — quick survey of all *_sf18.pt files in a directory.

For each file: position count, meta info (source, min_elo, draw rate),
whether outcome_values is present. Used to decide which files to combine
into a new training corpus.

Usage
-----
    python3 src/inspect_datasets.py /dtu/blackhole/0b/206091
    python3 src/inspect_datasets.py /dtu/blackhole/0b/206091 --pattern '*_sf18.pt'
    python3 src/inspect_datasets.py /dtu/blackhole/0b/206091 --exclude 2021_06_high low_elo
"""

import argparse
import glob
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="Directory containing the .pt files")
    ap.add_argument("--pattern", default="*_sf18.pt")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Substrings — any file whose name contains one is skipped")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.directory, args.pattern)))
    if not paths:
        print(f"No files match {args.pattern} in {args.directory}")
        sys.exit(1)

    print(f"{'name':<46}  {'N':>11}  {'min_elo':>8}  {'draws':>7}  {'outcome':>7}  {'cum':>11}")
    print("-" * 100)

    total = 0
    qualifying = 0
    for p in paths:
        name = os.path.basename(p)

        skipped_for = next((x for x in args.exclude if x in name), None)
        if skipped_for:
            print(f"{name:<46}  -- skipped (matched '{skipped_for}')")
            continue

        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"{name:<46}  ERROR: {e}")
            continue

        n_train = len(d["train"].get("fens", []))
        n_val   = len(d["val"].get("fens", []))
        n_total = n_train + n_val

        meta    = d.get("meta", {})
        min_elo = meta.get("min_elo", meta.get("elo_min", meta.get("elo", "?")))

        ov_train = d["train"].get("outcome_values")
        if ov_train is not None:
            ov = torch.cat([d["train"]["outcome_values"],
                            d["val"]["outcome_values"]])
            draw_pct = 100 * (ov.float().abs() < 1e-4).float().mean().item()
            has_outcome = "YES"
            total += n_total
            qualifying += 1
        else:
            draw_pct = float("nan")
            has_outcome = "NO"

        cum_str = f"{total:>11,}" if has_outcome == "YES" else "-"
        elo_str = str(min_elo)
        draw_str = f"{draw_pct:>5.1f}%" if has_outcome == "YES" else "  -"
        print(f"{name:<46}  {n_total:>11,}  {elo_str:>8}  {draw_str:>7}  "
              f"{has_outcome:>7}  {cum_str}")

    print("-" * 100)
    print(f"Files with outcome_values: {qualifying}    Total positions: {total:,}")


if __name__ == "__main__":
    main()
