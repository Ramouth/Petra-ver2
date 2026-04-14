"""
zigzag.py — Orchestrates the self-play → SF re-label → train loop.

Each round:
  1. SELF-PLAY   selfplay.py   → data/selfplay_r{n}.pt
  2. SF RELABEL  reeval_stockfish.py  → data/selfplay_r{n}_sf.pt
  3. TRAIN       train.py      → models/zigzag/r{n}/best.pt
  4. EVALUATE    evaluate.py   → gate check (>55% vs MCTS(material))

Round parameters (from ZIGZAG.md):
  Round | n_sim | SF depth | n_games | lr
    1   |   40  |    12    |   500   | 5e-4
    2   |   80  |    15    |   500   | 3e-4
    3   |  160  |    18    |   500   | 1e-4
    4   |  320  |    20    |   500   | 5e-5

CPU-only note: n_sim=320 is expensive on CPU. Cap at round 3 unless
GPU access is available. Use --rounds 1-3 for HPC CPU runs.

Usage
-----
    # Full run from round 1 (uses models/sf/best.pt as round-0 seed)
    python3 zigzag.py

    # Prototype: 50 games, start from round 1
    python3 zigzag.py --games 50 --workers 4 --rounds 1

    # Resume from a specific round
    python3 zigzag.py --start-round 2

    # Dry-run: print commands without executing
    python3 zigzag.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Round curriculum  (n_sim, sf_depth, n_games, lr)
# ---------------------------------------------------------------------------

ROUNDS = [
    (40,   12, 500, 5e-4),   # round 1
    (80,   15, 500, 3e-4),   # round 2
    (160,  18, 500, 1e-4),   # round 3
    (320,  20, 500, 5e-5),   # round 4
]

GATE_WINRATE  = 0.55   # new model must beat MCTS(material) at this rate
GATE_N_GAMES  = 200    # games for the gate evaluation (100 is too noisy at 55% threshold)
GATE_N_SIM    = 50     # n_sim for gate evaluation — enough search for value head signal

# Smoke-test parameters — tiny scale, just verifies the pipeline is wired correctly.
# Run locally before any HPC submission: python3 zigzag.py --smoke
SMOKE_N_GAMES    = 10
SMOKE_N_SIM      = 10
SMOKE_SF_DEPTH   = 5
SMOKE_EPOCHS     = 2
SMOKE_GATE_GAMES = 20
SMOKE_GATE_N_SIM = 10

SRC_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(SRC_DIR)
DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models", "zigzag")
SEED_MODEL    = os.path.join(PROJECT_ROOT, "models", "sf", "best.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], dry_run: bool) -> int:
    """Print and optionally execute a command. Returns exit code."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    t0 = time.time()
    result = subprocess.run(cmd)
    print(f"  (exit {result.returncode}, {time.time()-t0:.0f}s)", flush=True)
    return result.returncode


_seed_model_override: str = None   # set by main() from --seed-model arg


def _model_for_round(r: int) -> str:
    """Path to the best model from round r (0 = supervised seed)."""
    if r == 0:
        return _seed_model_override or SEED_MODEL
    return os.path.join(MODELS_DIR, f"r{r}", "best.pt")


# ---------------------------------------------------------------------------
# Round execution
# ---------------------------------------------------------------------------

def _build_train_cmd(sf_pt, out_dir, lr, epochs, init_model,
                     anchor_dataset, anchor_frac, seed=42):
    """Build the train.py command for one seed."""
    cmd = [
        sys.executable, os.path.join(SRC_DIR, "train.py"),
        "--dataset",    sf_pt,
        "--out",        out_dir,
        "--lr",         str(lr),
        "--epochs",     str(epochs),
        "--patience",   "3",
        "--seed",       str(seed),
        "--init-model", init_model,
    ]
    if anchor_dataset:
        cmd += ["--anchor-dataset", anchor_dataset,
                "--anchor-frac",    str(anchor_frac)]
    return cmd


def _train_with_seed_selection(sf_pt, out_dir, lr, epochs, init_model,
                                anchor_dataset, anchor_frac, n_seeds, dry_run):
    """
    Train n_seeds candidates in parallel, select winner by geometry gap.

    If n_seeds == 1, runs a single training job directly into out_dir.
    If n_seeds > 1, runs N jobs in parallel into out_dir/seed{N} subdirs,
    then calls select_geometry.py to copy the winner to out_dir/best.pt.

    Returns True on success, False on any failure.
    """
    if n_seeds == 1:
        cmd = _build_train_cmd(sf_pt, out_dir, lr, epochs, init_model,
                               anchor_dataset, anchor_frac, seed=42)
        rc = _run(cmd, dry_run)
        return rc == 0

    # Multiple seeds — parallel training
    print(f"\n--- Training {n_seeds} seeds in parallel ---")
    cand_dirs = [os.path.join(out_dir, f"seed{s}") for s in range(n_seeds)]
    for d in cand_dirs:
        os.makedirs(d, exist_ok=True)

    procs = []
    for s, cand_dir in enumerate(cand_dirs):
        cmd = _build_train_cmd(sf_pt, cand_dir, lr, epochs, init_model,
                               anchor_dataset, anchor_frac, seed=s)
        print(f"\n$ {' '.join(cmd)}  [seed {s}, background]", flush=True)
        if not dry_run:
            procs.append(subprocess.Popen(cmd))

    if not dry_run:
        print(f"\nWaiting for {n_seeds} training jobs ...", flush=True)
        exit_codes = [p.wait() for p in procs]
        for s, rc in enumerate(exit_codes):
            print(f"  seed {s}: exit {rc}")
        if any(rc != 0 for rc in exit_codes):
            print("  One or more seeds failed. Aborting round.")
            return False

    # Select best geometry among candidates
    candidates = [os.path.join(d, "best.pt") for d in cand_dirs]
    rc = _run([
        sys.executable, os.path.join(SRC_DIR, "select_geometry.py"),
        "--candidates", *candidates,
        "--dataset",    sf_pt,
        "--out",        os.path.join(out_dir, "best.pt"),
    ], dry_run)
    # Exit 2 = geometry flat (warning, not fatal — proceed but flag it)
    if rc not in (0, 2):
        print(f"  select_geometry.py failed (exit {rc}). Aborting round.")
        return False
    if rc == 2:
        print("  WARNING: all seeds have flat geometry (gap < 0.02). "
              "Proceeding with best available, but this is a bad sign.")
    return True


def run_round(r: int, n_sim: int, sf_depth: int, n_games: int, lr: float,
              workers: int, dry_run: bool,
              anchor_dataset: str = None, anchor_frac: float = 0.15,
              opening_book: str = None,
              epochs: int = 15,
              gate_n_games: int = GATE_N_GAMES,
              gate_n_sim: int = GATE_N_SIM,
              n_seeds: int = 1,
              stockfish: str = None) -> bool:
    """
    Execute one full zigzag round. Returns True if gate passed.
    """
    print(f"\n{'='*60}")
    print(f"  ROUND {r}  |  n_sim={n_sim}  sf_depth={sf_depth}  "
          f"n_games={n_games}  lr={lr:.0e}")
    print(f"{'='*60}")

    prev_model  = _model_for_round(r - 1)
    selfplay_pt = os.path.join(DATA_DIR, f"selfplay_r{r}.pt")
    sf_pt       = os.path.join(DATA_DIR, f"selfplay_r{r}_sf.pt")
    out_dir     = os.path.join(MODELS_DIR, f"r{r}")

    os.makedirs(DATA_DIR,  exist_ok=True)
    os.makedirs(out_dir,   exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

    # --- 1. Self-play ---
    print(f"\n--- Step 1: Self-play ({n_games} games, n_sim={n_sim}) ---")
    selfplay_cmd = [
        sys.executable, os.path.join(SRC_DIR, "selfplay.py"),
        "--model",   prev_model,
        "--games",   str(n_games),
        "--n-sim",   str(n_sim),
        "--out",     selfplay_pt,
        "--workers", str(workers),
    ]
    if opening_book:
        selfplay_cmd += ["--opening-book", opening_book]
    rc = _run(selfplay_cmd, dry_run)
    if rc != 0:
        print(f"  Self-play failed (exit {rc}). Aborting round.")
        return False

    # --- 2. SF re-label ---
    print(f"\n--- Step 2: Stockfish re-label (depth {sf_depth}) ---")
    reeval_cmd = [
        sys.executable, os.path.join(SRC_DIR, "reeval_stockfish.py"),
        "--dataset", selfplay_pt,
        "--out",     sf_pt,
        "--depth",   str(sf_depth),
        "--workers", str(workers),
    ]
    if stockfish:
        reeval_cmd += ["--stockfish", stockfish]
    rc = _run(reeval_cmd, dry_run)
    if rc != 0:
        print(f"  SF re-label failed (exit {rc}). Aborting round.")
        return False

    # --- 3. Train ---
    seed_note = f"{n_seeds} seeds → select_geometry" if n_seeds > 1 else "1 seed"
    print(f"\n--- Step 3: Train (lr={lr:.0e}, epochs={epochs}, {seed_note}) ---")
    ok = _train_with_seed_selection(
        sf_pt=sf_pt, out_dir=out_dir, lr=lr, epochs=epochs,
        init_model=prev_model,
        anchor_dataset=anchor_dataset, anchor_frac=anchor_frac,
        n_seeds=n_seeds, dry_run=dry_run,
    )
    if not ok:
        print(f"  Training step failed. Aborting round.")
        return False

    # --- 3b. Geometry probe ---
    # Run automatically after every training step so geometry health is
    # visible in the HPC logs without manual intervention.
    print(f"\n--- Step 3b: Geometry probe ---")
    _run([
        sys.executable, os.path.join(SRC_DIR, "probe_geometry.py"),
        "--model",   os.path.join(out_dir, "best.pt"),
        "--dataset", sf_pt,
    ], dry_run)
    # Non-zero exit from probe is non-fatal — it's diagnostic, not a gate.

    # --- 4. Gate evaluation ---
    print(f"\n--- Step 4: Gate evaluation ({gate_n_games} games, n_sim={gate_n_sim}) ---")
    new_model = _model_for_round(r)
    rc = _run([
        sys.executable, os.path.join(SRC_DIR, "evaluate.py"),
        "--model",      new_model,
        "--step",       "5",
        "--games",      str(gate_n_games),
        "--n-sim",      str(gate_n_sim),
        "--workers",    str(workers),
        "--temp-moves", "10",
    ], dry_run)
    # evaluate.py writes its own output to stdout — capture to log as well
    # (for now just check the return code; a non-zero means GATE FAILED)
    if dry_run:
        print(f"  [dry-run] Gate assumed passed.")
        return True

    passed = rc == 0
    print(f"\n  Round {r} gate: {'PASSED' if passed else 'FAILED'}")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds",      type=int, default=None,
                    help="How many rounds to run (default: all 4)")
    ap.add_argument("--start-round", type=int, default=1,
                    help="Resume from this round (default: 1)")
    ap.add_argument("--games",       type=int, default=None,
                    help="Override n_games for all rounds (prototype: 50)")
    ap.add_argument("--workers",     type=int, default=1,
                    help="Parallel self-play workers")
    ap.add_argument("--opening-book", type=str, default=None,
                    help="Path to FEN file (gen_openings.py output) for self-play seeding.")
    ap.add_argument("--sf-anchor",   type=str, default=None,
                    help="Path to supervised SF dataset mixed into each training round "
                         "as a collapse anchor (e.g. dataset_sf.pt). Recommended: 15%%.")
    ap.add_argument("--anchor-frac", type=float, default=0.15,
                    help="Fraction of self-play train size to sample from anchor (default: 0.15)")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Print commands without executing")
    ap.add_argument("--n-seeds",     type=int, default=1,
                    help="Seeds to train in parallel per round, selecting best geometry "
                         "(default: 1). Use 3 for round 1 cold starts — essentially free "
                         "on multi-core HPC and converts initialisation variance into an "
                         "advantage. Subsequent rounds can use 1 (fine-tuning from prior round).")
    ap.add_argument("--seed-model",  type=str, default=None,
                    help="Path to round-0 seed model (default: models/sf/best.pt). "
                         "Override when the supervised model is saved elsewhere.")
    ap.add_argument("--stockfish",   type=str, default=None,
                    help="Path to Stockfish binary for SF re-label step. "
                         "Defaults to reeval_stockfish.py default (/usr/games/stockfish). "
                         "Must be set on HPC where Stockfish is not in /usr/games/.")
    ap.add_argument("--smoke",       action="store_true",
                    help="Smoke test: run round 1 at tiny scale to verify pipeline wiring. "
                         f"Uses {SMOKE_N_GAMES} games, n_sim={SMOKE_N_SIM}, "
                         f"SF depth {SMOKE_SF_DEPTH}, {SMOKE_EPOCHS} epochs, "
                         f"{SMOKE_GATE_GAMES} gate games. "
                         "Run this locally before submitting any HPC job.")
    args = ap.parse_args()

    global _seed_model_override
    if args.seed_model:
        _seed_model_override = args.seed_model

    # --smoke: override round 1 with minimal params, exit after one round.
    if args.smoke:
        print(f"\n{'='*60}")
        print(f"  SMOKE TEST  —  verifying pipeline wiring at tiny scale")
        print(f"  games={SMOKE_N_GAMES}  n_sim={SMOKE_N_SIM}  sf_depth={SMOKE_SF_DEPTH}")
        print(f"  epochs={SMOKE_EPOCHS}  gate_games={SMOKE_GATE_GAMES}")
        print(f"  This is NOT a real training run — results are meaningless.")
        print(f"{'='*60}")

        prev_model = _model_for_round(0)
        if not os.path.exists(prev_model):
            print(f"\nERROR: seed model not found: {prev_model}")
            print("  Smoke test requires models/sf/best.pt (run supervised training first).")
            sys.exit(1)

        _, _, _, lr = ROUNDS[0]   # use round 1 LR
        passed = run_round(
            r=1, n_sim=SMOKE_N_SIM, sf_depth=SMOKE_SF_DEPTH,
            n_games=SMOKE_N_GAMES, lr=lr,
            workers=args.workers, dry_run=False,
            anchor_dataset=args.sf_anchor,
            anchor_frac=args.anchor_frac,
            opening_book=args.opening_book,
            epochs=SMOKE_EPOCHS,
            gate_n_games=SMOKE_GATE_GAMES,
            gate_n_sim=SMOKE_GATE_N_SIM,
            n_seeds=1,   # always single seed in smoke test
        )
        if passed:
            print(f"\nSmoke test complete. Pipeline is wired correctly.")
            print(f"Gate result is meaningless at this scale — ignore win rate.")
        else:
            print(f"\nSmoke test gate FAILED — but this may be expected at {SMOKE_N_GAMES} games.")
            print(f"Check the output above for errors (missing files, crashes, wrong paths).")
            print(f"A gate fail here is not a problem — a crash or file-not-found is.")
        sys.exit(0)

    start = args.start_round
    end   = start + (args.rounds or len(ROUNDS)) - 1
    end   = min(end, len(ROUNDS))

    print(f"zigzag.py  |  rounds {start}–{end}  |  workers={args.workers}"
          + ("  [DRY RUN]" if args.dry_run else ""))

    for r in range(start, end + 1):
        n_sim, sf_depth, n_games, lr = ROUNDS[r - 1]
        if args.games:
            n_games = args.games

        prev_model = _model_for_round(r - 1)
        if not args.dry_run and not os.path.exists(prev_model):
            print(f"\nERROR: seed model not found: {prev_model}")
            print("  Round 0 requires models/sf/best.pt (run supervised training first).")
            sys.exit(1)

        passed = run_round(
            r=r, n_sim=n_sim, sf_depth=sf_depth,
            n_games=n_games, lr=lr,
            workers=args.workers, dry_run=args.dry_run,
            anchor_dataset=args.sf_anchor,
            anchor_frac=args.anchor_frac,
            opening_book=args.opening_book,
            n_seeds=args.n_seeds,
            stockfish=args.stockfish,
        )

        if not passed:
            print(f"\nRound {r} gate FAILED. Stopping zigzag.")
            print("Investigate: check geometry probe, val R², and self-play game quality.")
            sys.exit(1)

        print(f"\nRound {r} complete. Model saved → {_model_for_round(r)}")

    print(f"\nZigzag complete. Ran rounds {start}–{end}.")


if __name__ == "__main__":
    main()
