"""
Split a SF-relabelled dataset into decisive and balanced subsets.

After SF re-evaluation, value labels are continuous (tanh of CP score), not
the discrete {+1, -0.1, -1} from Lichess game outcomes.  This script partitions
the dataset by |value| so each training stage works on a focused signal:

  decisive.pt  — |value| ≥ decisive_thresh (default 0.5)
                  Strong win/loss positions → reinforce win/loss axis.

  balanced.pt  — |value| < balanced_thresh  (default 0.3)
                  Near-equal / drawn positions → push draw dimension orthogonal.

Positions in the ambiguous band [balanced_thresh, decisive_thresh) are dropped;
they carry mixed signal and do not cleanly belong to either stage.

Both output files preserve the train/val split, meta dict, and all keys
(tensors, values, move_idxs, legal_masks, visit_dists if present).

Usage
-----
    python3 src/split_dataset.py \\
        --dataset data/dataset_new_sf.pt \\
        --decisive-out data/decisive.pt \\
        --balanced-out data/balanced.pt

    # Custom thresholds:
    python3 src/split_dataset.py \\
        --dataset data/dataset_new_sf.pt \\
        --decisive-thresh 0.6 \\
        --balanced-thresh 0.25
"""

import argparse
import os
import sys

import torch


def _filter_split(split: dict, mask: torch.Tensor) -> dict:
    """Return a new split dict containing only rows where mask is True."""
    idx = mask.nonzero(as_tuple=True)[0]
    out = {}
    for k, v in split.items():
        out[k] = v[idx]
    return out


def split_dataset(
    dataset_path: str,
    decisive_out: str,
    balanced_out: str,
    decisive_thresh: float = 0.5,
    balanced_thresh: float = 0.3,
) -> None:
    print(f"Loading {dataset_path} ...")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)

    for split_name in ("train", "val"):
        split = data[split_name]
        values = split["values"]
        abs_v  = values.abs()

        decisive_mask = abs_v >= decisive_thresh
        balanced_mask = abs_v <  balanced_thresh

        n_total    = len(values)
        n_decisive = decisive_mask.sum().item()
        n_balanced = balanced_mask.sum().item()
        n_dropped  = n_total - n_decisive - n_balanced

        print(f"  {split_name}: {n_total:,} total → "
              f"{n_decisive:,} decisive (|v|≥{decisive_thresh}) | "
              f"{n_balanced:,} balanced (|v|<{balanced_thresh}) | "
              f"{n_dropped:,} dropped ({balanced_thresh}–{decisive_thresh} band)")

        data[f"{split_name}_decisive"] = _filter_split(split, decisive_mask)
        data[f"{split_name}_balanced"] = _filter_split(split, balanced_mask)

    # Build two separate dataset dicts — same structure as the original.
    meta = data.get("meta", {})

    decisive_data = {
        "train": data["train_decisive"],
        "val":   data["val_decisive"],
        "meta": {
            **meta,
            "n_train": len(data["train_decisive"]["tensors"]),
            "n_val":   len(data["val_decisive"]["tensors"]),
            "split":   f"decisive (|v|≥{decisive_thresh})",
        },
    }
    balanced_data = {
        "train": data["train_balanced"],
        "val":   data["val_balanced"],
        "meta": {
            **meta,
            "n_train": len(data["train_balanced"]["tensors"]),
            "n_val":   len(data["val_balanced"]["tensors"]),
            "split":   f"balanced (|v|<{balanced_thresh})",
        },
    }

    print(f"\nSaving decisive → {decisive_out}  "
          f"({decisive_data['meta']['n_train']:,} train / "
          f"{decisive_data['meta']['n_val']:,} val)")
    torch.save(decisive_data, decisive_out)

    print(f"Saving balanced → {balanced_out}  "
          f"({balanced_data['meta']['n_train']:,} train / "
          f"{balanced_data['meta']['n_val']:,} val)")
    torch.save(balanced_data, balanced_out)

    print("\nDone.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",        required=True,
                    help="Input SF-relabelled .pt dataset")
    ap.add_argument("--decisive-out",   default=None,
                    help="Output path for decisive subset (default: <stem>_decisive.pt)")
    ap.add_argument("--balanced-out",   default=None,
                    help="Output path for balanced subset (default: <stem>_balanced.pt)")
    ap.add_argument("--decisive-thresh", type=float, default=0.5,
                    help="|value| ≥ this → decisive (default: 0.5)")
    ap.add_argument("--balanced-thresh", type=float, default=0.3,
                    help="|value| <  this → balanced (default: 0.3)")
    args = ap.parse_args()

    stem = os.path.splitext(args.dataset)[0]
    decisive_out = args.decisive_out or f"{stem}_decisive.pt"
    balanced_out = args.balanced_out or f"{stem}_balanced.pt"

    split_dataset(
        dataset_path=args.dataset,
        decisive_out=decisive_out,
        balanced_out=balanced_out,
        decisive_thresh=args.decisive_thresh,
        balanced_thresh=args.balanced_thresh,
    )


if __name__ == "__main__":
    main()
