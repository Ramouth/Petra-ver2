"""
select_geometry.py — A/B geometry selection for zigzag cold starts.

Runs probe_geometry.py on multiple candidate models and selects the one
with the best geometry — measured by the separation gap (Check 2), not
val loss. Copies the winner to a target path for use in the zigzag loop.

Why geometry quality, not val loss
------------------------------------
Val loss measures value prediction accuracy. Geometry quality (separation
gap) measures whether the bottleneck has learned a structured representation
where winning positions and losing positions occupy distinct regions.

These are correlated but not identical. A model with slightly worse val loss
but a cleaner geometry is strictly better for this project — the geometry
is what we're building, and the geometry is what Gaussa will inherit.

Why A/B selection
-----------------
A cold start on new architecture has real variance. Two runs with identical
hyperparameters can land in very different geometry basins. A/B selection
converts that variance from a risk into an advantage: run 2-3 candidates
in parallel (essentially free on HPC) and keep the best geometry.

Usage
-----
    # Train 3 candidates (run in parallel on HPC)
    python3 src/train.py --dataset data/sf_r1.pt --out models/cand_s0 --seed 0 &
    python3 src/train.py --dataset data/sf_r1.pt --out models/cand_s1 --seed 1 &
    python3 src/train.py --dataset data/sf_r1.pt --out models/cand_s2 --seed 2 &
    wait

    # Select best geometry and copy to zigzag target
    python3 src/select_geometry.py \\
        --candidates models/cand_s0/best.pt models/cand_s1/best.pt models/cand_s2/best.pt \\
        --dataset    data/sf_r1.pt \\
        --out        models/zigzag/r1/best.pt

    # With learning rate variants instead of seeds
    python3 src/select_geometry.py \\
        --candidates models/cand_lr3/best.pt models/cand_lr5/best.pt models/cand_lr7/best.pt \\
        --dataset    data/sf_r1.pt \\
        --out        models/zigzag/r1/best.pt \\
        --n 3000

Selection criteria (in order)
------------------------------
1. Effective rank > 20     — primary (geometry uses multiple dimensions)
2. Separation gap > 0.02   — secondary (geometry is structured along an axis)
3. Val loss                — tiebreaker

Why rank before gap: a collapsed 1D geometry can have a perfect separation gap
(all wins at one pole, all losses at the other) while effective rank = 1. That
is a degenerate geometry — the MCTS value function collapses to a scalar. Rank
ensures we select models with distributed geometry, not just a lucky axis.

If all candidates have rank < 20: warns and picks best available,
but signals that something may be architecturally wrong.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Probe a single candidate
# ---------------------------------------------------------------------------

def probe_candidate(model_path: str, dataset_path: str, n: int) -> dict:
    """
    Run probe_geometry.py on one model. Returns parsed metrics dict.

    Keys: separation_gap, effective_rank, raw_output, error
    """
    print(f"\n  Probing: {model_path}")
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(SRC_DIR, "probe_geometry.py"),
            "--model",   model_path,
            "--dataset", dataset_path,
            "--n",       str(n),
        ],
        capture_output=True,
        text=True,
    )

    metrics = {
        "model_path":     model_path,
        "separation_gap": None,
        "effective_rank": None,
        "betti_1":        None,
        "raw_output":     result.stdout,
        "error":          result.returncode != 0,
    }

    if result.returncode != 0:
        print(f"    ERROR (exit {result.returncode})")
        print(result.stderr[:200] if result.stderr else "(no stderr)")
        return metrics

    # Parse metrics from probe output
    for line in result.stdout.splitlines():
        if "Separation gap:" in line:
            try:
                metrics["separation_gap"] = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        if "Effective rank:" in line:
            try:
                metrics["effective_rank"] = float(line.split(":")[1].split("/")[0].strip())
            except (ValueError, IndexError):
                pass
        if "Topology:" in line and "β1=" in line:
            try:
                import re
                m = re.search(r"β1=(\d+)", line)
                if m:
                    metrics["betti_1"] = int(m.group(1))
            except (ValueError, AttributeError):
                pass

    gap  = metrics["separation_gap"]
    rank = metrics["effective_rank"]
    b1   = metrics["betti_1"]
    b1_str = f"  β1={b1}" if b1 is not None else ""
    print(f"    separation_gap={gap:.4f}  effective_rank={rank:.1f}{b1_str}" if gap is not None
          else "    Could not parse metrics")

    return metrics


# ---------------------------------------------------------------------------
# Select winner
# ---------------------------------------------------------------------------

def select_winner(results: list) -> dict:
    """
    Pick the best candidate by: gap → effective_rank → (tiebreaker: first).

    Returns the winning result dict.
    """
    # Filter out errored probes
    valid = [r for r in results if not r["error"] and r["separation_gap"] is not None]

    if not valid:
        print("\nERROR: No valid probe results. Cannot select a winner.")
        sys.exit(1)

    # Sort: effective_rank → separation_gap → β1 (tiebreaker: loops = structure).
    valid.sort(key=lambda r: (
        r["effective_rank"]  if r["effective_rank"]  is not None else 0,
        r["separation_gap"]  if r["separation_gap"]  is not None else -999,
        1 if (r["betti_1"] or 0) > 0 else 0,   # β1 > 0 preferred
    ), reverse=True)

    return valid[0]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list, winner: dict):
    print("\n" + "="*60)
    print("A/B GEOMETRY SELECTION REPORT")
    print("="*60)
    print(f"\n  {'Model':<45}  {'Gap':>7}  {'Rank':>6}  {'β1':>4}  {'Status'}")
    print(f"  {'-'*45}  {'-------':>7}  {'------':>6}  {'----':>4}  ------")

    for r in results:
        gap  = f"{r['separation_gap']:.4f}" if r['separation_gap'] is not None else "ERROR"
        rank = f"{r['effective_rank']:.1f}" if r['effective_rank'] is not None else "  —"
        b1   = str(r['betti_1']) if r['betti_1'] is not None else "?"
        is_winner = (r["model_path"] == winner["model_path"])
        status = "← WINNER" if is_winner else ""
        name = os.path.basename(os.path.dirname(r["model_path"]))
        print(f"  {name:<45}  {gap:>7}  {rank:>6}  {b1:>4}  {status}")

    gap  = winner["separation_gap"]
    rank = winner["effective_rank"]
    print(f"\n  Winner: {winner['model_path']}")
    print(f"  Rank:   {rank:.1f}  Gap: {gap:.4f}", end="")

    if rank is not None and rank < 20:
        print(f"\n\n  WARNING: Best effective rank ({rank:.1f}) < 20 — geometry may be collapsed.")
        print(f"  Consider: different LR variants, more training epochs,")
        print(f"  or check that the doover architecture fixes are applied.")
    elif gap < 0.02:
        print(f"\n\n  WARNING: Best gap ({gap:.4f}) < 0.02 — geometry may be flat.")
        print(f"  Consider: different LR variants, more training epochs.")
    elif gap < 0.05:
        print(f"  (weak — geometry forming but not strong)")
    elif gap < 0.10:
        print(f"  (moderate — proceed to gate evaluation)")
    else:
        print(f"  (good — geometry is structured)")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Select best geometry from multiple trained candidates."
    )
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="Paths to candidate model .pt files")
    ap.add_argument("--dataset",    required=True,
                    help="Path to dataset .pt for geometry probing")
    ap.add_argument("--out",        required=True,
                    help="Output path — winner is copied here")
    ap.add_argument("--n",          type=int, default=2000,
                    help="Number of val positions to probe per candidate (default: 2000)")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Print what would happen without copying")
    args = ap.parse_args()

    print(f"select_geometry.py")
    print(f"  candidates : {len(args.candidates)}")
    print(f"  dataset    : {args.dataset}")
    print(f"  out        : {args.out}")
    print(f"  n          : {args.n}")

    # Validate candidates exist
    for path in args.candidates:
        if not os.path.exists(path):
            print(f"\nERROR: candidate not found: {path}")
            sys.exit(1)

    if not os.path.exists(args.dataset):
        print(f"\nERROR: dataset not found: {args.dataset}")
        sys.exit(1)

    # Probe all candidates
    print(f"\nProbing {len(args.candidates)} candidates ...\n")
    results = [probe_candidate(c, args.dataset, args.n) for c in args.candidates]

    # Select winner
    winner = select_winner(results)

    # Report
    print_report(results, winner)

    # Copy winner to output path
    if args.dry_run:
        print(f"  [dry-run] Would copy → {args.out}")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        shutil.copy2(winner["model_path"], args.out)
        print(f"  Copied winner → {args.out}")

    # Exit non-zero if geometry is flat — lets HPC scripts detect failure
    if winner["separation_gap"] < 0.02:
        sys.exit(2)   # 2 = flat geometry (distinct from 1 = script error)


if __name__ == "__main__":
    main()
