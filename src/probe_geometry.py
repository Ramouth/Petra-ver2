"""
Geometry space probe for PetraNet.

Checks whether the 128-dim bottleneck has learned meaningful structure:

  1. Eigenvalue distribution  — is variance spread or collapsed?
  2. Win/draw/loss clustering  — do positions with different labels separate?
  3. Known position probes     — do structurally extreme positions (KQ vs K,
                                 equal endgame, opening) land in different regions?
  4. Nearest-neighbour sanity  — do the 5 nearest neighbours of a probe position
                                 share its label more than random?

Usage
-----
    python3 src/probe_geometry.py --model models/best.pt --dataset dataset.pt
    python3 src/probe_geometry.py --model models/best.pt --dataset dataset.pt --n 5000
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

def load_geometry_vectors(model: PetraNet, dataset_path: str, n: int = 5000):
    """
    Load n positions from the val set, run through geometry(), return
    (vectors, values) as numpy arrays.

    Values are kept STM-relative (side-to-move perspective) — consistent
    with the board_to_tensor encoding which is always from STM's view.
    Do NOT convert to White-absolute: after board flipping, the geometry
    vectors are STM-relative and the labels must match.
    """
    data   = torch.load(dataset_path, map_location="cpu", weights_only=False)
    split  = data["val"]
    tensors = split["tensors"][:n].float()   # (N, 14, 8, 8) uint8 → float32
    values  = split["values"][:n].numpy()    # (N,)  STM-relative — keep as-is

    model.eval()
    batch_size = 256
    vecs = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i+batch_size].to(device)
            g = model.geometry(batch)
            vecs.append(g.cpu().numpy())

    vecs = np.concatenate(vecs, axis=0)   # (N, 128)
    return vecs, values


def _label_class(v: float, strict: bool = False) -> str:
    """
    Bucket a STM-relative value into win/draw/loss.

    strict=False (default): ±0.5 thresholds — works well for discrete ±1
      outcome labels (old supervised datasets).

    strict=True: ±0.7 win/loss, ±0.3 draw, with a "unclear" middle band
      discarded.  Use for continuous SF-labelled datasets (e.g.
      dataset_feb_sf.pt) where many positions cluster in the 0.3–0.5
      range and inflate the "draw" bucket with near-wins/losses.
    """
    if strict:
        if v >  0.7: return "win"
        if v < -0.7: return "loss"
        if abs(v) < 0.3: return "draw"
        return "unclear"
    if v > 0.5:  return "win"
    if v < -0.5: return "loss"
    return "draw"


def board_to_geo(model: PetraNet, board: chess.Board) -> np.ndarray:
    t = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.geometry(t).cpu().numpy()[0]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def alignment(win_vecs: np.ndarray, loss_vecs: np.ndarray,
              alpha: float = 2.0) -> float:
    """
    Alignment loss: mean pairwise distance within class.
    Lower = better aligned (same-class vectors are close).
    Wang & Isola (2020), alpha=2.
    """
    def pairwise_alignment(vecs):
        rng = np.random.default_rng(42)
        n = len(vecs)
        n_pairs = min(500, n * (n - 1) // 2)
        i_idx = rng.integers(0, n, n_pairs)
        j_idx = rng.integers(0, n, n_pairs)
        mask = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        diffs = vecs[i_idx] - vecs[j_idx]
        return float(np.mean(np.linalg.norm(diffs, axis=1) ** alpha))

    return (pairwise_alignment(win_vecs) + pairwise_alignment(loss_vecs)) / 2


def uniformity(vecs: np.ndarray, t: float = 2.0) -> float:
    """
    Uniformity loss: log mean pairwise Gaussian kernel.
    Lower = more uniformly distributed on the sphere.
    Wang & Isola (2020), t=2.
    """
    rng = np.random.default_rng(42)
    n = len(vecs)
    n_pairs = min(1000, n * (n - 1) // 2)
    i_idx = rng.integers(0, n, n_pairs)
    j_idx = rng.integers(0, n, n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    sq_dists = np.sum((vecs[i_idx] - vecs[j_idx]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-t * sq_dists))) + 1e-8)


# ---------------------------------------------------------------------------
# Check 1: Eigenvalue distribution
# ---------------------------------------------------------------------------

def check_eigenvalue_distribution(vecs: np.ndarray):
    print("\n" + "="*60)
    print("CHECK 1 — Eigenvalue distribution")
    print("="*60)

    # Mean-centre
    centred = vecs - vecs.mean(axis=0)
    cov = np.cov(centred.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    total   = eigvals.sum() + 1e-8

    top1_pct  = 100 * eigvals[0]  / total
    top5_pct  = 100 * eigvals[:5].sum()  / total
    top10_pct = 100 * eigvals[:10].sum() / total

    print(f"  Top-1  eigenvalue: {top1_pct:.1f}% of variance")
    print(f"  Top-5  eigenvalues: {top5_pct:.1f}% of variance")
    print(f"  Top-10 eigenvalues: {top10_pct:.1f}% of variance")

    # Effective rank — how many dimensions are actually being used.
    # After L2 normalisation all norms are 1.0, so mean norm is uninformative.
    # Effective rank = (sum eigenvalues)^2 / sum(eigenvalues^2)
    # Healthy: > 30. Collapsed: < 10.
    effective_rank = (eigvals.sum() ** 2) / ((eigvals ** 2).sum() + 1e-8)
    print(f"  Effective rank: {effective_rank:.1f} / 128  (healthy > 30, collapsed < 10)")

    # Diagnosis
    if top1_pct > 50:
        print(f"\n  COLLAPSED — top-1 dim dominates ({top1_pct:.1f}%)")
        print(f"  (Phase 0 collapsed encoder had top-1 = 77.8%)")
    elif top1_pct > 25:
        print(f"\n  PARTIAL — some concentration but not collapsed")
    else:
        print(f"\n  HEALTHY — variance spread across dimensions")

    return eigvals


# ---------------------------------------------------------------------------
# Check 2: Win / draw / loss separation
# ---------------------------------------------------------------------------

def check_label_separation(vecs: np.ndarray, values: np.ndarray):
    print("\n" + "="*60)
    print("CHECK 2 — Win / draw / loss separation")
    print("="*60)

    # Broad buckets (±0.5): comparable across all dataset types
    labels = np.array([_label_class(v) for v in values])
    # Strict buckets (±0.7/±0.3): better for continuous SF labels —
    # removes the ambiguous 0.3–0.7 band that inflates the draw count
    # and drags win·draw cosine up artificially.
    strict_labels = np.array([_label_class(v, strict=True) for v in values])
    n_unclear = (strict_labels == "unclear").sum()

    win_vecs  = vecs[labels == "win"]
    loss_vecs = vecs[labels == "loss"]
    draw_vecs = vecs[labels == "draw"]

    print(f"  Samples (broad ±0.5)  — win: {len(win_vecs)}, draw: {len(draw_vecs)}, "
          f"loss: {len(loss_vecs)}")
    win_vecs_s  = vecs[strict_labels == "win"]
    loss_vecs_s = vecs[strict_labels == "loss"]
    draw_vecs_s = vecs[strict_labels == "draw"]
    print(f"  Samples (strict ±0.7) — win: {len(win_vecs_s)}, draw: {len(draw_vecs_s)}, "
          f"loss: {len(loss_vecs_s)}  ({n_unclear} unclear discarded)")

    if len(win_vecs) == 0 or len(loss_vecs) == 0:
        print("  SKIPPED — insufficient win/loss samples for separation check")
        return

    # Centroids (broad buckets — for cross-run comparability)
    c_win  = win_vecs.mean(axis=0)
    c_loss = loss_vecs.mean(axis=0)
    c_draw = draw_vecs.mean(axis=0) if len(draw_vecs) > 0 else None

    win_loss_sim  = cosine_sim(c_win, c_loss)
    print(f"\n  Centroid cosine similarities (broad ±0.5 buckets):")
    print(f"    win  · loss  = {win_loss_sim:.4f}  (lower = better separation)")
    if c_draw is not None:
        print(f"    win  · draw  = {cosine_sim(c_win,  c_draw):.4f}")
        print(f"    loss · draw  = {cosine_sim(c_loss, c_draw):.4f}")

    # Repeat with strict buckets — removes the 0.3–0.7 ambiguous band
    if len(win_vecs_s) > 0 and len(loss_vecs_s) > 0 and len(draw_vecs_s) > 0:
        cs_win  = win_vecs_s.mean(axis=0)
        cs_loss = loss_vecs_s.mean(axis=0)
        cs_draw = draw_vecs_s.mean(axis=0)
        print(f"  Centroid cosine similarities (strict ±0.7/±0.3 buckets):")
        print(f"    win  · loss  = {cosine_sim(cs_win, cs_loss):.4f}")
        print(f"    win  · draw  = {cosine_sim(cs_win,  cs_draw):.4f}  ← draw-dimension signal")
        print(f"    loss · draw  = {cosine_sim(cs_loss, cs_draw):.4f}")

    # Within-class vs between-class distances (sample 500 per class)
    rng = np.random.default_rng(42)
    n_sample = min(500, len(win_vecs), len(loss_vecs))
    w = win_vecs[rng.choice(len(win_vecs),   n_sample, replace=False)]
    l = loss_vecs[rng.choice(len(loss_vecs), n_sample, replace=False)]

    within_win  = np.mean([cosine_sim(w[i], w[j])
                           for i, j in zip(range(0, n_sample, 2), range(1, n_sample, 2))])
    within_loss = np.mean([cosine_sim(l[i], l[j])
                           for i, j in zip(range(0, n_sample, 2), range(1, n_sample, 2))])
    between     = np.mean([cosine_sim(w[i], l[i]) for i in range(n_sample)])

    print(f"\n  Within-class cosine similarity (higher = tighter cluster):")
    print(f"    win  pairs: {within_win:.4f}")
    print(f"    loss pairs: {within_loss:.4f}")
    print(f"  Between-class (win vs loss): {between:.4f}")
    print(f"  Separation gap: {((within_win + within_loss) / 2 - between):.4f}")

    gap = (within_win + within_loss) / 2 - between
    if gap > 0.05:
        print(f"\n  GOOD — win/loss positions occupy distinct regions")
    elif gap > 0.01:
        print(f"\n  WEAK — some separation but small")
    else:
        print(f"\n  NONE — win/loss positions are not separated")

    # Alignment + uniformity (Wang & Isola 2020)
    align = alignment(win_vecs, loss_vecs)
    unif  = uniformity(np.vstack([win_vecs, loss_vecs]))
    print(f"\n  Alignment  : {align:.4f}  (lower = tighter same-class clusters)")
    print(f"  Uniformity : {unif:.4f}  (lower = better spread across sphere)")
    print(f"  Note: both should improve together — one without the other is a warning sign")

    return c_win, c_loss


# ---------------------------------------------------------------------------
# Check 3: Known position probes
# ---------------------------------------------------------------------------

def check_known_positions(model: PetraNet, c_win: np.ndarray, c_loss: np.ndarray):
    print("\n" + "="*60)
    print("CHECK 3 — Known position probes")
    print("="*60)

    probes = [
        ("Starting position",           chess.Board()),
        ("KQ vs K — White to move",     chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")),
        ("KQ vs K — Black to move",     chess.Board("4k3/8/8/8/8/8/8/4K2Q b - - 0 1")),
        ("Equal endgame (KR vs KR)",    chess.Board("8/3k4/8/r7/8/8/3K4/7R w - - 0 1")),
        ("White queen up",              chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")),
        ("Black queen up",              chess.Board("4K3/8/8/8/8/8/8/q3k3 w - - 0 1")),
        ("Complex middlegame",          chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")),
    ]

    print(f"  {'Position':<35}  {'→win':>6}  {'→loss':>6}  {'value':>7}  label")
    print(f"  {'-'*35}  {'------':>6}  {'------':>6}  {'-------':>7}  -----")

    for name, board in probes:
        g   = board_to_geo(model, board)
        sw  = cosine_sim(g, c_win)
        sl  = cosine_sim(g, c_loss)
        val = model.value(board, device)
        dominant = "win" if sw > sl else "loss"
        print(f"  {name:<35}  {sw:>6.3f}  {sl:>6.3f}  {val:>+7.3f}  → {dominant}")


# ---------------------------------------------------------------------------
# Check 4: Structural drawness
# ---------------------------------------------------------------------------

def check_drawness(model: PetraNet):
    print("\n" + "="*60)
    print("CHECK 4 — Structural drawness")
    print("="*60)

    probes = [
        ("KR vs KR structural draw",
         chess.Board("8/3k4/8/r7/8/8/3K4/7R w - - 0 1"),
         ">0.7"),
        ("KNN vs K theoretical draw",
         chess.Board("8/8/8/8/4k3/8/4NN2/4K3 w - - 0 1"),
         ">0.7"),
        ("Sharp balanced middlegame",
         chess.Board("rnbqkbnr/pp1ppppp/8/2p5/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"),
         "<0.3"),
        ("KQ vs K decisive",
         chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1"),
         "<0.3"),
    ]

    passed = 0
    for name, board, gate in probes:
        d_val = model.drawness(board, device)
        v_val = model.value(board, device)
        if gate == ">0.7":
            ok = d_val > 0.7
        else:
            ok = d_val < 0.3
        passed += int(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"  {mark:<4} {name:<32} drawness={d_val:.3f}  "
              f"value={v_val:+.3f}  gate {gate}")

    print(f"\n  Drawness gates passed: {passed}/{len(probes)}")
    if passed == len(probes):
        print("  GOOD — structural draws separate from balanced/decisive positions")
    elif passed >= 2:
        print("  WEAK — drawness signal exists but is not reliable yet")
    else:
        print("  NONE — drawness head is untrained or collapsed")


# ---------------------------------------------------------------------------
# Check 5: Nearest-neighbour label consistency
# ---------------------------------------------------------------------------

def check_nearest_neighbours(vecs: np.ndarray, values: np.ndarray, k: int = 5, n_probe: int = 200):
    print("\n" + "="*60)
    print("CHECK 4 — Nearest-neighbour label consistency")
    print("="*60)
    print(f"  For {n_probe} random positions: are the {k} nearest neighbours")
    print(f"  more likely to share the same label than random chance?")

    labels = np.array([_label_class(v) for v in values])
    rng    = np.random.default_rng(42)
    idxs   = rng.choice(len(vecs), n_probe, replace=False)

    # Normalise for cosine similarity
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    normed = vecs / norms

    match_rates = []
    for idx in idxs:
        sims     = normed @ normed[idx]
        sims[idx] = -1.0   # exclude self
        top_k    = np.argsort(sims)[-k:]
        same     = sum(labels[j] == labels[idx] for j in top_k)
        match_rates.append(same / k)

    mean_match = np.mean(match_rates)

    # Baseline: random label match rate (win/draw/loss only — no invalid)
    label_counts = {l: (labels == l).sum() for l in ["win", "loss", "draw"]}
    n = len(labels)
    baseline = sum((c / n) ** 2 for c in label_counts.values() if c > 0)

    print(f"\n  Mean NN label match rate : {mean_match:.3f}")
    print(f"  Random baseline          : {baseline:.3f}")
    print(f"  Lift                     : {mean_match - baseline:+.3f}")

    if mean_match - baseline > 0.1:
        print(f"\n  GOOD — nearest neighbours share labels more than chance")
    elif mean_match - baseline > 0.02:
        print(f"\n  WEAK — slight tendency, not strong")
    else:
        print(f"\n  NONE — geometry does not cluster by label")


# ---------------------------------------------------------------------------
# Check 6: Equal-region subspace — structural draws vs balanced
# ---------------------------------------------------------------------------

def _gen_structural_draws(n_per_type: int = 150, seed: int = 42):
    """
    Generate valid structural draw positions: KR vs KR, KNN vs K, KB vs KB
    (same-colour bishops). Returns list of (label, chess.Board).
    """
    rng = np.random.default_rng(seed)
    positions = []
    all_sq = list(range(64))

    def _sq_color(sq):
        return (chess.square_file(sq) + chess.square_rank(sq)) % 2

    def _try_board(piece_map, turn):
        board = chess.Board(fen=None)
        board.clear()
        for sq, pc in piece_map.items():
            board.set_piece_at(sq, pc)
        board.turn = chess.WHITE if turn else chess.BLACK
        try:
            return board if board.is_valid() else None
        except Exception:
            return None

    # KR vs KR
    attempts = 0
    while sum(1 for n, _ in positions if n == "KRvKR") < n_per_type and attempts < 20000:
        attempts += 1
        sq = rng.choice(all_sq, 4, replace=False).tolist()
        wk, bk, wr, br = sq
        if chess.square_distance(wk, bk) <= 1:
            continue
        board = _try_board({
            wk: chess.Piece(chess.KING,  chess.WHITE),
            bk: chess.Piece(chess.KING,  chess.BLACK),
            wr: chess.Piece(chess.ROOK,  chess.WHITE),
            br: chess.Piece(chess.ROOK,  chess.BLACK),
        }, rng.random() < 0.5)
        if board is not None:
            positions.append(("KRvKR", board))

    # KNN vs K  (two white knights vs lone black king — theoretical draw)
    attempts = 0
    while sum(1 for n, _ in positions if n == "KNNvK") < n_per_type and attempts < 20000:
        attempts += 1
        sq = rng.choice(all_sq, 4, replace=False).tolist()
        wk, bk, wn1, wn2 = sq
        if chess.square_distance(wk, bk) <= 1:
            continue
        board = _try_board({
            wk:  chess.Piece(chess.KING,   chess.WHITE),
            bk:  chess.Piece(chess.KING,   chess.BLACK),
            wn1: chess.Piece(chess.KNIGHT, chess.WHITE),
            wn2: chess.Piece(chess.KNIGHT, chess.WHITE),
        }, rng.random() < 0.5)
        if board is not None:
            positions.append(("KNNvK", board))

    # KB vs KB  (same-colour bishops — draw)
    attempts = 0
    while sum(1 for n, _ in positions if n == "KBvKB") < n_per_type and attempts < 20000:
        attempts += 1
        sq = rng.choice(all_sq, 4, replace=False).tolist()
        wk, bk, wb, bb = sq
        if chess.square_distance(wk, bk) <= 1:
            continue
        if _sq_color(wb) != _sq_color(bb):   # must be same colour
            continue
        board = _try_board({
            wk: chess.Piece(chess.KING,   chess.WHITE),
            bk: chess.Piece(chess.KING,   chess.BLACK),
            wb: chess.Piece(chess.BISHOP, chess.WHITE),
            bb: chess.Piece(chess.BISHOP, chess.BLACK),
        }, rng.random() < 0.5)
        if board is not None:
            positions.append(("KBvKB", board))

    return positions


def check_equal_subspace(model: PetraNet, vecs: np.ndarray, values: np.ndarray,
                         dataset_path: str):
    """
    Ask whether the |v|<0.2 region already geometrically separates structural
    draws from balanced positions.

    Group A — structural draws : generated KR vs KR, KNN vs K, KB vs KB.
    Group B — equal region     : probe dataset positions with |v|<0.2.
                                 Split by outcome_values if available:
                                 decisive-game positions are the sharpest
                                 negatives (equal-looking but proved not drawn).

    Outcome guide
    -------------
    accuracy > 85% : geometry already separates them. Frozen-backbone
                     drawness head can read it out — no backbone training needed.
    accuracy 60-85%: weak separation. Small contrastive training may suffice.
    accuracy ~50%  : no separation. Backbone geometry must change first.
    """
    print("\n" + "="*60)
    print("CHECK 6 — Equal-region subspace: structural draws vs balanced")
    print("="*60)
    print("  Question: does the |v|<0.2 geometry already separate")
    print("  KR vs KR / KNN vs K / KB vs KB from balanced positions?")

    # --- Group A: structural draw positions ---------------------------------
    draw_positions = _gen_structural_draws(n_per_type=150)
    if not draw_positions:
        print("  SKIPPED — could not generate structural draw positions")
        return

    model.eval()
    draw_vecs = []
    with torch.no_grad():
        for _, board in draw_positions:
            t = board_to_tensor(board).unsqueeze(0).to(device)
            g = model.geometry(t).cpu().numpy()[0]
            draw_vecs.append(g)
    draw_vecs = np.array(draw_vecs)

    type_counts = {}
    for name, _ in draw_positions:
        type_counts[name] = type_counts.get(name, 0) + 1
    type_str = "  ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))

    # --- Group B: equal-region positions from dataset -----------------------
    eq_mask = np.abs(values) < 0.2
    eq_vecs_all = vecs[eq_mask]

    # Try to load outcome_values to separate draw-game from decisive-game
    outcome_values = None
    try:
        data = torch.load(dataset_path, map_location="cpu", weights_only=False)
        split = data["val"]
        if "outcome_values" in split:
            ov = split["outcome_values"][:len(values)].numpy()
            outcome_values = ov[eq_mask]
    except Exception:
        pass

    if outcome_values is not None:
        dec_mask  = np.abs(outcome_values) > 0.5
        eq_dec    = eq_vecs_all[dec_mask]   # equal-looking, game was decisive
        eq_draw   = eq_vecs_all[~dec_mask]  # equal-looking, game was drawn
        print(f"\n  Group A — structural draws : {len(draw_vecs)} ({type_str})")
        print(f"  Group B — equal region     : {len(eq_vecs_all)} positions (|v|<0.2)")
        print(f"    of which decisive-game   : {len(eq_dec)}  ← sharpest negatives")
        print(f"    of which drawn-game      : {len(eq_draw)}")
        # Use decisive-game subset as the primary comparison if large enough
        eq_vecs = eq_dec if len(eq_dec) >= 50 else eq_vecs_all
        neg_label = "equal+decisive" if len(eq_dec) >= 50 else "equal (all)"
    else:
        eq_vecs   = eq_vecs_all
        neg_label = "equal (all)"
        print(f"\n  Group A — structural draws : {len(draw_vecs)} ({type_str})")
        print(f"  Group B — {neg_label:<18}: {len(eq_vecs)} positions (|v|<0.2)")
        print(f"  (outcome_values not in dataset — cannot split by game outcome)")

    if len(eq_vecs) < 20:
        print("  SKIPPED — too few equal-region positions in dataset")
        return

    # --- Centroid separation ------------------------------------------------
    c_draw  = draw_vecs.mean(axis=0)
    c_equal = eq_vecs.mean(axis=0)
    centroid_cos  = cosine_sim(c_draw, c_equal)
    between_dist  = float(np.linalg.norm(c_draw - c_equal))
    draw_spread   = float(np.linalg.norm(draw_vecs  - c_draw,  axis=1).mean())
    equal_spread  = float(np.linalg.norm(eq_vecs    - c_equal, axis=1).mean())

    print(f"\n  Centroid cosine (structural vs {neg_label}): {centroid_cos:+.4f}")
    print(f"  Between-centroid L2 distance              : {between_dist:.3f}")
    print(f"  Within-group spread  — structural {draw_spread:.3f}  "
          f"{neg_label} {equal_spread:.3f}")

    # --- PCA of the combined equal region -----------------------------------
    all_eq = np.vstack([draw_vecs, eq_vecs])
    centred = all_eq - all_eq.mean(axis=0)
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    pc1 = eigvecs[:, 0]
    pc2 = eigvecs[:, 1]
    d1, e1 = draw_vecs @ pc1, eq_vecs @ pc1
    d2, e2 = draw_vecs @ pc2, eq_vecs @ pc2
    var1 = 100 * eigvals[0] / (eigvals.sum() + 1e-8)
    var2 = 100 * eigvals[1] / (eigvals.sum() + 1e-8)
    sep_d = abs(d1.mean() - e1.mean()) / ((d1.std() + e1.std()) / 2 + 1e-8)

    print(f"\n  PCA of combined equal-region:")
    print(f"    PC1 {var1:.1f}%  PC2 {var2:.1f}% of variance")
    print(f"    PC1 projection — structural: {d1.mean():+.3f}±{d1.std():.3f}"
          f"  {neg_label}: {e1.mean():+.3f}±{e1.std():.3f}")
    print(f"    PC2 projection — structural: {d2.mean():+.3f}±{d2.std():.3f}"
          f"  {neg_label}: {e2.mean():+.3f}±{e2.std():.3f}")
    print(f"    Cohen's d on PC1: {sep_d:.3f}  (>1.0 = well separated along PC1)")

    # --- Linear separability ------------------------------------------------
    rng = np.random.default_rng(42)
    n_use = min(len(draw_vecs), len(eq_vecs), 250)
    A = draw_vecs[rng.choice(len(draw_vecs), n_use, replace=False)]
    B = eq_vecs[rng.choice(len(eq_vecs),     n_use, replace=False)]
    X = np.vstack([A, B]).astype(np.float32)
    y = np.array([1] * n_use + [0] * n_use)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        clf   = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        acc   = float(cv_scores.mean())
        print(f"\n  Logistic regression (5-fold CV, n={2*n_use}): {acc:.3f}")

        clf.fit(X, y)
        w_norm = clf.coef_[0] / (np.linalg.norm(clf.coef_[0]) + 1e-8)
        top5   = np.argsort(np.abs(w_norm))[-5:][::-1]
        print(f"  Top-5 separating dims: {top5.tolist()}")

    except ImportError:
        # Nearest-centroid fallback
        correct = sum(
            np.linalg.norm(v - c_draw) < np.linalg.norm(v - c_equal) for v in A
        ) + sum(
            np.linalg.norm(v - c_equal) < np.linalg.norm(v - c_draw) for v in B
        )
        acc = correct / (2 * n_use)
        print(f"\n  Nearest-centroid accuracy (sklearn absent): {acc:.3f}")

    print()
    if acc > 0.85:
        print("  SEPARABLE   — frozen backbone drawness head should work.")
    elif acc > 0.60:
        print("  WEAK        — partial separation; contrastive training may help.")
    else:
        print("  NOT SEPARABLE — backbone must change before drawness head can learn.")


# ---------------------------------------------------------------------------
# Check 7: Topological health
# ---------------------------------------------------------------------------

def check_topology(model: PetraNet, val_loader, n_sample: int = 300):
    print("\n" + "="*60)
    print("CHECK 5 — Topological health (β0, β1, persistence entropy)")
    print("="*60)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from topology_monitor import topological_health_check, format_topology_line

    topo = topological_health_check(model, val_loader, epoch=999,
                                    n_sample=n_sample, device=device)
    print(format_topology_line(topo))
    b1 = topo.get("betti_1")
    if b1 is not None:
        if b1 > 0:
            print(f"\n  GOOD — β1={b1} loops detected: geometry has non-trivial structure")
        else:
            print(f"\n  FLAT — β1=0: geometry may be a linear manifold (no loops)")
    return topo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True, help="Path to model .pt")
    ap.add_argument("--dataset", required=True, help="Path to dataset .pt")
    ap.add_argument("--n",       type=int, default=5000,
                    help="Number of val positions to probe (default 5000)")
    args = ap.parse_args()

    model = PetraNet().to(device)
    sd = torch.load(args.model, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  New/untrained parameters in model definition: {missing}")
    if unexpected:
        print(f"  Unexpected checkpoint parameters ignored: {unexpected}")
    model.eval()
    print(f"Loaded model from {args.model}")

    print(f"Extracting geometry vectors for {args.n} val positions ...")
    vecs, values = load_geometry_vectors(model, args.dataset, n=args.n)
    print(f"  Done. Shape: {vecs.shape}")

    check_eigenvalue_distribution(vecs)
    c_win, c_loss = check_label_separation(vecs, values)
    check_known_positions(model, c_win, c_loss)
    check_drawness(model)
    check_nearest_neighbours(vecs, values)
    check_equal_subspace(model, vecs, values, args.dataset)

    # Build a minimal val_loader for the topology check
    from torch.utils.data import DataLoader, TensorDataset
    data   = torch.load(args.dataset, map_location="cpu", weights_only=False)
    split  = data["val"]
    t_val  = split["tensors"][:args.n].float()
    v_val  = split["values"][:args.n]
    ds_val = TensorDataset(t_val, v_val)
    val_loader = DataLoader(ds_val, batch_size=256, shuffle=False)
    check_topology(model, val_loader)

    print("\n" + "="*60)
    print("Probe complete.")
    print("="*60)


if __name__ == "__main__":
    main()
