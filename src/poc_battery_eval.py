#!/usr/bin/env python3
"""
poc_battery_eval.py — run the drawness gate on a PetraNet model.

Loads the PoC battery (built by build_poc_battery.py) and asks: from the
position alone, can the model distinguish 'drawn' from 'decisive' games
when the engine value is dead-even (|SF| < 0.2)?

This is the project's PoC novelty: no engine separates "balanced but
drifting toward a result" from "structurally drawn." The gate measures
whether the learned geometry encodes that distinction.

Three independent signals are reported, each evaluated via 5-fold CV on
the battery (strict train/test disjointness — fits and tests are never
on the same position):

  (a) cosine-to-centroid : sign(cos(g, centroid_drawn) − cos(g, centroid_decisive))
  (b) PC1 projection     : project on first principal component of fold-train
                           geometry, oriented so drawn > 0
  (c) logistic regression: fit LR on fold-train, score fold-test

Metrics per signal:
  - balanced accuracy with 95% binomial CI
  - AUC
  - Cohen's d on signal gap between classes

Random baseline = 50% balanced accuracy. The gate "passes" if all three
signals are above chance with 95% CI excluding 0.5.

Usage
-----
    python3 src/poc_battery_eval.py \
        --model   models/natural_v2/best.pt \
        --battery data/poc_battery_v0.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PetraNet
from config import device


def parse_args():
    ap = argparse.ArgumentParser(description="Drawness PoC gate eval.")
    ap.add_argument("--model",       required=True, help="Path to model best.pt")
    ap.add_argument("--battery",     required=True, help="poc_battery_v*.pt")
    ap.add_argument("--n-folds",     type=int, default=5)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--batch-size",  type=int, default=256)
    return ap.parse_args()


# ─── Geometry extraction ──────────────────────────────────────────────────

def compute_geometry(model: PetraNet, tensors: torch.Tensor,
                     batch_size: int) -> np.ndarray:
    """Forward all positions through the bottleneck. Returns (N, 128)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i+batch_size].float().to(device)
            g = model.geometry(batch).cpu().numpy()
            out.append(g)
    return np.concatenate(out, axis=0)


# ─── Signal definitions ───────────────────────────────────────────────────

def signal_centroid(g_train: np.ndarray, y_train: np.ndarray,
                    g_test: np.ndarray) -> np.ndarray:
    """
    Higher score = more drawn-like.

    score = cos(g, centroid_drawn) − cos(g, centroid_decisive)
    """
    c_drawn    = g_train[y_train == 0].mean(axis=0)
    c_decisive = g_train[y_train == 1].mean(axis=0)

    def _norm(v):
        return v / (np.linalg.norm(v) + 1e-8)

    c_d = _norm(c_drawn)
    c_w = _norm(c_decisive)
    g_test_n = g_test / (np.linalg.norm(g_test, axis=1, keepdims=True) + 1e-8)
    return g_test_n @ c_d - g_test_n @ c_w


def signal_pc1(g_train: np.ndarray, y_train: np.ndarray,
               g_test: np.ndarray) -> np.ndarray:
    """
    Higher score = more drawn-like.

    Compute PC1 on fold-train geometry, project test, orient sign so
    drawn-class has higher mean projection.
    """
    centred = g_train - g_train.mean(axis=0)
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, np.argmax(eigvals)]

    proj_train = g_train @ pc1
    if proj_train[y_train == 0].mean() < proj_train[y_train == 1].mean():
        pc1 = -pc1
    return g_test @ pc1


def signal_logreg(g_train: np.ndarray, y_train: np.ndarray,
                  g_test: np.ndarray) -> np.ndarray:
    """
    Higher score = more drawn-like.

    score = P(drawn | g) from logistic regression fit on fold-train.
    """
    from sklearn.linear_model import LogisticRegression
    # y=1 means decisive in our encoding — flip to make "drawn" the positive
    # class so the returned probability aligns with "higher = drawn".
    y_drawn = (y_train == 0).astype(int)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(g_train, y_drawn)
    return clf.predict_proba(g_test)[:, 1]


# ─── Metrics ──────────────────────────────────────────────────────────────

def balanced_accuracy(scores: np.ndarray, y: np.ndarray) -> tuple:
    """
    Threshold at 0 (signed scores) or 0.5 (probabilities). For consistency,
    threshold at the median of fold-train scores in the calling code.
    Here we threshold at 0 by convention — caller has already centred LR
    scores.

    Returns (balanced_accuracy, n_drawn, n_decisive, tpr, tnr).
    """
    pred_drawn = scores > 0
    is_drawn   = y == 0
    is_dec     = y == 1
    tpr = (pred_drawn & is_drawn).sum() / max(is_drawn.sum(), 1)   # drawn correct
    tnr = (~pred_drawn & is_dec).sum() / max(is_dec.sum(), 1)      # decisive correct
    return 0.5 * (tpr + tnr), int(is_drawn.sum()), int(is_dec.sum()), tpr, tnr


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """AUC where positive class = drawn (label 0). Higher score = more drawn."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score((y == 0).astype(int), scores))


def cohens_d(scores: np.ndarray, y: np.ndarray) -> float:
    a = scores[y == 0]
    b = scores[y == 1]
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def binomial_ci(p: float, n: int, z: float = 1.96) -> tuple:
    """95% Wilson interval (more honest than normal approx near edges)."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half   = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ─── CV runner ────────────────────────────────────────────────────────────

def run_cv(geometry: np.ndarray, labels: np.ndarray, n_folds: int,
           seed: int) -> dict:
    """
    Stratified K-fold CV. For each signal, accumulate test-fold scores
    across all folds (each position scored exactly once, by a model that
    didn't see it during fit). Then compute aggregate metrics.
    """
    rng = np.random.default_rng(seed)
    n   = len(labels)

    # Stratified shuffle: split drawn and decisive separately into n_folds.
    drawn_idx    = np.where(labels == 0)[0]
    decisive_idx = np.where(labels == 1)[0]
    rng.shuffle(drawn_idx)
    rng.shuffle(decisive_idx)
    fold_drawn    = np.array_split(drawn_idx,    n_folds)
    fold_decisive = np.array_split(decisive_idx, n_folds)

    signals = ["centroid", "pc1", "logreg"]
    score_funcs = {
        "centroid": signal_centroid,
        "pc1":      signal_pc1,
        "logreg":   signal_logreg,
    }
    out_scores = {s: np.zeros(n, dtype=np.float64) for s in signals}

    for k in range(n_folds):
        test_idx = np.concatenate([fold_drawn[k], fold_decisive[k]])
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        g_tr = geometry[train_idx]; y_tr = labels[train_idx]
        g_te = geometry[test_idx]

        for s in signals:
            scores = score_funcs[s](g_tr, y_tr, g_te)
            # Centre scores so threshold at 0 means "more drawn-like than
            # the fold-train median". Required for the balanced-acc metric.
            tr_scores = score_funcs[s](g_tr, y_tr, g_tr)
            out_scores[s][test_idx] = scores - np.median(tr_scores)

    return out_scores


# ─── Reporting ────────────────────────────────────────────────────────────

def report(name: str, scores: np.ndarray, labels: np.ndarray):
    bacc, n_d, n_w, tpr, tnr = balanced_accuracy(scores, labels)
    a   = auc(scores, labels)
    d   = cohens_d(scores, labels)
    n_total = n_d + n_w
    lo, hi = binomial_ci(bacc, n_total)
    pass_ = "PASS" if lo > 0.5 else "FAIL"
    print(f"  {name:<10} bacc={bacc:.4f}  CI=[{lo:.4f}, {hi:.4f}]  "
          f"AUC={a:.4f}  d={d:+.3f}  TPR={tpr:.3f} TNR={tnr:.3f}  [{pass_}]")


def main():
    args = parse_args()

    print(f"Device: {device}")
    print(f"Loading model: {args.model}")
    model = PetraNet().to(device)
    sd = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)

    print(f"Loading battery: {args.battery}")
    battery = torch.load(args.battery, map_location="cpu", weights_only=False)
    tensors = battery["tensors"]
    labels  = battery["labels"].numpy().astype(np.int64)
    print(f"  N={len(labels):,}  drawn={(labels==0).sum():,}  "
          f"decisive={(labels==1).sum():,}")
    print(f"  meta: {battery['meta']}")

    print(f"\nComputing geometry vectors ...")
    geometry = compute_geometry(model, tensors, args.batch_size)
    print(f"  shape={geometry.shape}")

    print(f"\nRunning {args.n_folds}-fold CV ...")
    scores = run_cv(geometry, labels, args.n_folds, args.seed)

    print(f"\n{'='*78}")
    print(f"DRAWNESS GATE — {os.path.basename(args.model)}")
    print(f"{'='*78}")
    print(f"  Battery: {os.path.basename(args.battery)}  N={len(labels):,}")
    print(f"  Random baseline = 0.500.  Gate passes if 95% CI lower-bound > 0.5.")
    print()
    print(f"  {'signal':<10} {'bacc':<6}  {'95% CI':<18}  {'AUC':<6}  "
          f"{'d':<7}  {'TPR / TNR':<14}")
    for s in ["centroid", "pc1", "logreg"]:
        report(s, scores[s], labels)
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
