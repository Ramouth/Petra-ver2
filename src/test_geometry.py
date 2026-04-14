"""
Geometry consistency tests for PetraNet.

Tests structural properties the geometry should satisfy if it is encoding
chess value correctly.

Tests
-----
  1. Material monotonicity   — adding stronger pieces moves geometry toward win
  2. Piece value ordering    — Q > R > B ≈ N > P in geometry projection
  3. STM symmetry            — board.mirror() (swap colors+STM) should give similar geometry
  4. Forced mate convergence — as mate approaches, win projection increases
  5. Transposition consistency — same position via different move orders = same geometry

Tests 1, 2, 4 project geometry onto a linear probe axis fitted on game outcomes
from the original dataset.pt. The model may be trained on SF centipawns, so game
outcomes are an orthogonal supervision signal — the probe is independent of the
value head. R² on the val set measures how much game-outcome signal the geometry
carries.

Tests 3, 5 are purely structural — they require no dataset.

Usage
-----
    python3 src/test_geometry.py --model models/best.pt --dataset data/dataset.pt
"""

import argparse
import os
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PetraNet
from board import board_to_tensor
from config import device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_existing_path(cli_value: str | None, candidates: list[str], label: str) -> str:
    """Return the explicit CLI path or the first existing default candidate."""
    if cli_value:
        return cli_value

    for path in candidates:
        if os.path.exists(path):
            return path

    joined = ", ".join(candidates)
    raise FileNotFoundError(
        f"Could not resolve {label}. Pass --{label} explicitly or create one of: {joined}"
    )

def geo(model: PetraNet, board: chess.Board) -> np.ndarray:
    t = board_to_tensor(board).unsqueeze(0).float().to(device)
    return model.geometry(t).cpu().numpy()[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def fit_outcome_probe(model: PetraNet, dataset_path: str,
                      batch_size: int = 512) -> tuple:
    """
    Fit a linear probe predicting game outcomes from geometry vectors.

    Uses the original dataset.pt (win/loss/draw labels), not the SF-reeval
    dataset. Since the model was trained on SF centipawns, game outcomes are
    an orthogonal supervision signal — the probe axis is independent of the
    value head weights.

    Returns (axis, bias, r2_train, r2_val).
      axis     — L2-normalised weight vector of the probe (128-dim)
      bias     — scalar intercept
      r2_train — R² on train split
      r2_val   — R² on val split
    """
    print(f"  Loading {dataset_path} ...")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)

    def _geoms_and_labels(split):
        tensors = data[split]["tensors"]        # (N, 14, 8, 8)
        values  = data[split]["values"].numpy() # (N,) game outcomes
        geoms   = []
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i+batch_size].float().to(device)
            g = model.geometry(batch).cpu().numpy()
            geoms.append(g)
        return np.vstack(geoms), values

    print("  Computing geometry vectors for train split ...")
    X_train, y_train = _geoms_and_labels("train")
    print("  Computing geometry vectors for val split ...")
    X_val,   y_val   = _geoms_and_labels("val")

    print(f"  Fitting ridge regression on {len(y_train):,} positions ...")
    lam = 1e-3
    XtX = X_train.T @ X_train + lam * np.eye(X_train.shape[1])
    w   = np.linalg.solve(XtX, X_train.T @ y_train)
    b   = float(y_train.mean() - (X_train.mean(axis=0) @ w))

    def r2(X, y):
        pred   = X @ w + b
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-8)

    r2_train = r2(X_train, y_train)
    r2_val   = r2(X_val,   y_val)

    axis = w / (np.linalg.norm(w) + 1e-8)
    return axis, b, r2_train, r2_val


def probe_proj(g: np.ndarray, axis: np.ndarray) -> float:
    """Scalar projection of a geometry vector onto the probe axis."""
    return float(np.dot(g, axis))


def color_flip(board: chess.Board) -> chess.Board:
    """Return the board with colors swapped (mirror position)."""
    return board.mirror()


PASS  = "✓ PASS"
FAIL  = "✗ FAIL"
WARN  = "~ WARN"


# ---------------------------------------------------------------------------
# Test 1: Material monotonicity
# ---------------------------------------------------------------------------

def check_material_monotonicity(model: PetraNet, axis: np.ndarray):
    print("\n" + "="*60)
    print("TEST 1 — Material monotonicity")
    print("  Adding stronger pieces should increase outcome-probe projection.")
    print("="*60)

    base = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")  # bare kings

    pieces = [
        (chess.PAWN,   chess.WHITE, chess.E4, "White pawn   (e4)"),
        (chess.KNIGHT, chess.WHITE, chess.D4, "White knight (d4)"),
        (chess.BISHOP, chess.WHITE, chess.C1, "White bishop (c1)"),
        (chess.ROOK,   chess.WHITE, chess.A1, "White rook   (a1)"),
        (chess.QUEEN,  chess.WHITE, chess.D1, "White queen  (d1)"),
    ]

    base_proj = probe_proj(geo(model, base), axis)
    print(f"\n  {'Position':<30}  {'probe':>8}  {'delta':>8}")
    print(f"  {'-'*30}  {'--------':>8}  {'--------':>8}")
    print(f"  {'Bare kings (baseline)':<30}  {base_proj:>+8.4f}  {'':>8}")

    projections = [base_proj]
    for piece_type, color, square, label in pieces:
        b = base.copy()
        b.set_piece_at(square, chess.Piece(piece_type, color))
        p = probe_proj(geo(model, b), axis)
        delta = p - base_proj
        projections.append(p)
        print(f"  {label:<30}  {p:>+8.4f}  {delta:>+8.4f}")

    passing = all(projections[i] < projections[i+1] for i in range(len(projections)-1))
    result = PASS if passing else FAIL
    print(f"\n  Result: {result}  (monotonically increasing: {passing})")
    return passing


# ---------------------------------------------------------------------------
# Test 2: Piece value ordering
# ---------------------------------------------------------------------------

def check_piece_value_ordering(model: PetraNet, axis: np.ndarray):
    print("\n" + "="*60)
    print("TEST 2 — Piece value ordering")
    print("  Outcome-probe projection should respect Q > R > B ≈ N > P.")
    print("="*60)

    base = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")

    piece_positions = [
        (chess.PAWN,   chess.WHITE, chess.E4, "Pawn"),
        (chess.KNIGHT, chess.WHITE, chess.D4, "Knight"),
        (chess.BISHOP, chess.WHITE, chess.C1, "Bishop"),
        (chess.ROOK,   chess.WHITE, chess.A1, "Rook"),
        (chess.QUEEN,  chess.WHITE, chess.D1, "Queen"),
    ]

    projs = {}
    print(f"\n  {'Piece':<10}  {'probe':>8}")
    print(f"  {'-'*10}  {'--------':>8}")
    for piece_type, color, square, label in piece_positions:
        b = base.copy()
        b.set_piece_at(square, chess.Piece(piece_type, color))
        p = probe_proj(geo(model, b), axis)
        projs[label] = p
        print(f"  {label:<10}  {p:>+8.4f}")

    checks = [
        ("Q > R",  projs["Queen"]  > projs["Rook"]),
        ("R > B",  projs["Rook"]   > projs["Bishop"]),
        ("R > N",  projs["Rook"]   > projs["Knight"]),
        ("B > P",  projs["Bishop"] > projs["Pawn"]),
        ("N > P",  projs["Knight"] > projs["Pawn"]),
    ]

    all_pass = True
    print()
    for label, ok in checks:
        status = PASS if ok else FAIL
        print(f"  {status}  {label}")
        if not ok:
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Test 3: STM symmetry
# ---------------------------------------------------------------------------

def check_stm_symmetry(model: PetraNet):
    print("\n" + "="*60)
    print("TEST 3 — STM symmetry")
    print("  board.mirror() swaps colors AND side to move.")
    print("  After board flipping, both positions look identical to the CNN:")
    print("  STM pieces at rank 0, same material advantage.")
    print("  Geometry vectors should be highly similar (cosine > 0.85).")
    print("="*60)

    positions = [
        ("KQ vs K (white winning)",  chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")),
        ("White up rook",            chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")),
        ("White up two pawns",       chess.Board("4k3/8/8/8/8/8/PPP5/4K3 w - - 0 1")),
        ("Equal (KR vs KR)",         chess.Board("r3k3/8/8/8/8/8/8/R3K3 w - - 0 1")),
        ("Starting position",        chess.Board()),
    ]

    print(f"\n  {'Position':<30}  {'cosine sim':>10}  result")
    print(f"  {'-'*30}  {'-'*10}  ------")

    results = []
    for label, board in positions:
        g      = geo(model, board)
        g_flip = geo(model, color_flip(board))
        # After board flipping: both tensors should be identical → geometry identical
        # With L2 norm: cosine_sim(g, g) = 1.0 for identical inputs
        cos = cosine_sim(g, g_flip)
        ok = cos > 0.85
        status = PASS if cos > 0.85 else (WARN if cos > 0.7 else FAIL)
        results.append(cos)
        print(f"  {label:<30}  {cos:>+10.4f}  {status}")

    mean_cos = np.mean(results)
    overall = PASS if mean_cos > 0.85 else (WARN if mean_cos > 0.7 else FAIL)
    print(f"\n  Mean cosine similarity: {mean_cos:+.4f}  →  {overall}")
    print(f"  (Perfect symmetry = 1.0, uncorrelated = 0.0)")
    print(f"  Note: untrained model may score 1.0 (identical random outputs).")
    print(f"        What matters is this stays high after training.")
    return mean_cos > 0.85


# ---------------------------------------------------------------------------
# Test 4: Forced mate convergence
# ---------------------------------------------------------------------------

def check_forced_mate_convergence(model: PetraNet, axis: np.ndarray):
    print("\n" + "="*60)
    print("TEST 4 — Forced mate convergence")
    print("  As mate approaches (KQ vs K), outcome-probe projection should increase.")
    print("="*60)

    positions = [
        ("Queen enters endgame",     chess.Board("8/8/8/8/4k3/8/8/Q3K3 w - - 0 1")),
        ("Queen closer",             chess.Board("8/8/8/4Q3/4k3/8/8/4K3 w - - 0 1")),
        ("King cooperating",         chess.Board("8/8/5Q2/8/3k4/4K3/8/8 w - - 0 1")),
        ("Near mate",                chess.Board("8/8/8/8/7k/6Q1/8/6K1 w - - 0 1")),
        ("Mate in 1",                chess.Board("7k/8/6KQ/8/8/8/8/8 w - - 0 1")),
    ]

    print(f"\n  {'Position':<28}  {'probe':>8}  {'value':>7}")
    print(f"  {'-'*28}  {'--------':>8}  {'-------':>7}")

    projections = []
    for label, board in positions:
        p = probe_proj(geo(model, board), axis)
        v = model.value(board, device)
        projections.append(p)
        print(f"  {label:<28}  {p:>+8.4f}  {v:>+7.3f}")

    violations = sum(projections[i] >= projections[i+1] for i in range(len(projections)-1))
    passing = violations <= 1
    result = PASS if passing else (WARN if violations <= 2 else FAIL)
    print(f"\n  Result: {result}  (non-monotone steps: {violations}/4)")
    return passing


# ---------------------------------------------------------------------------
# Test 5: Transposition consistency
# ---------------------------------------------------------------------------

def check_transposition_consistency(model: PetraNet):
    print("\n" + "="*60)
    print("TEST 5 — Transposition consistency")
    print("  Same position via different move orders should give same geometry.")
    print("="*60)

    transpositions = [
        ("Italian Game", [
            # Via 1.e4 e5 2.Nf3 Nc6 3.Bc4
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
            # Via 1.e4 e5 2.Bc4 Nc6 3.Nf3
            ["e2e4", "e7e5", "f1c4", "b8c6", "g1f3"],
        ]),
        ("Symmetric pawn structure", [
            ["e2e4", "d7d5", "d2d4", "e7e6"],
            ["d2d4", "e7e6", "e2e4", "d7d5"],
        ]),
        ("Piece development", [
            ["g1f3", "g8f6", "b1c3", "b8c6"],
            ["b1c3", "b8c6", "g1f3", "g8f6"],
        ]),
    ]

    print(f"\n  {'Transposition':<25}  {'cosine sim':>10}  {'max |Δg|':>9}  result")
    print(f"  {'-'*25}  {'-'*10}  {'-'*9}  ------")

    all_pass = True
    for name, move_seqs in transpositions:
        boards = []
        for moves in move_seqs:
            b = chess.Board()
            for m in moves:
                b.push_uci(m)
            boards.append(b)

        # Check all pairs reach the same FEN (otherwise it's not a transposition)
        fens = [b.fen().split(" ")[0] for b in boards]  # position only, ignore side/castling
        if len(set(fens)) > 1:
            print(f"  {name:<25}  {'N/A — not same pos':>21}  SKIP")
            continue

        geos = [geo(model, b) for b in boards]
        cos  = cosine_sim(geos[0], geos[1])
        max_delta = float(np.max(np.abs(geos[0] - geos[1])))
        ok = cos > 0.99
        status = PASS if ok else (WARN if cos > 0.95 else FAIL)
        if not ok:
            all_pass = False
        print(f"  {name:<25}  {cos:>+10.4f}  {max_delta:>9.4f}  {status}")

    return all_pass


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: dict):
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total  = len(results)
    passed = sum(results.values())
    print(f"\n  Passed: {passed}/{total}\n")
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default=None,
                    help="Path to model .pt (defaults to models/best.pt if present)")
    ap.add_argument("--dataset", default=None,
                    help="Path to original dataset.pt with game outcome labels "
                         "(defaults to data/dataset.pt if present)")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        model_path = _resolve_existing_path(
            args.model,
            [os.path.join(repo_root, "models", "best.pt")],
            "model",
        )
        dataset_path = _resolve_existing_path(
            args.dataset,
            [os.path.join(repo_root, "data", "dataset.pt")],
            "dataset",
        )
    except FileNotFoundError as e:
        ap.error(str(e))

    model = PetraNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {model_path}")

    print("\nFitting game-outcome linear probe on geometry vectors ...")
    axis, bias, r2_train, r2_val = fit_outcome_probe(model, dataset_path)
    print(f"  Probe R²  train={r2_train:.4f}  val={r2_val:.4f}")
    print(f"  (R² > 0.1 suggests geometry carries meaningful game-outcome signal)")

    results = {}
    results["Material monotonicity"]     = check_material_monotonicity(model, axis)
    results["Piece value ordering"]      = check_piece_value_ordering(model, axis)
    results["STM symmetry"]              = check_stm_symmetry(model)
    results["Forced mate convergence"]   = check_forced_mate_convergence(model, axis)
    results["Transposition consistency"] = check_transposition_consistency(model)

    print_summary(results)


if __name__ == "__main__":
    main()
