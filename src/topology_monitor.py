"""
topology_monitor.py — Topological health monitoring during training.

Tracks β0 (connected components), β1 (loops), and persistence entropy
across training epochs. Provides early abort signal when topology is
collapsing rather than forming.

The key insight: separation gap tells you IF geometry is forming.
Topology tells you WHAT SHAPE it's forming into — and whether that
shape will be useful for Gaussa.

A healthy geometry trajectory:
  β0:  high → drops → 1      (convergence to connected space)
  β1:  0 → rises → stabilises (loops = structure)
  H:   low → rises → plateau  (increasing topological complexity)

A collapsing geometry trajectory (looks identical in separation gap):
  β0:  high → drops → 1      (same as healthy early on)
  β1:  0 → tiny blip → 0     (structure never takes hold)
  H:   low → tiny rise → 0   (complexity collapses)
"""

import numpy as np
import torch
from typing import Optional


def extract_geometry_sample(model, val_loader, n: int = 300,
                             device=None) -> np.ndarray:
    """
    Extract n geometry vectors from the val set.
    Lightweight — designed for mid-training calls.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    vecs = []
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            if count >= n:
                break
            tensors = batch[0].float().to(device)
            remaining = n - count
            tensors = tensors[:remaining]
            g = model.geometry(tensors)
            vecs.append(g.cpu().numpy())
            count += len(tensors)

    return np.concatenate(vecs, axis=0)[:n]


def topological_health_check(model, val_loader, epoch: int,
                              persistence_threshold: float = 0.05,
                              n_sample: int = 300,
                              device=None) -> dict:
    """
    Lightweight topology check designed to run during training.

    persistence_threshold: minimum loop lifetime to count as structure
        (shorter-lived loops are noise). Start at 0.05, tune if needed.

    Returns dict with betti_0, betti_1, entropy, verdict.
    """
    try:
        from ripser import ripser
    except ImportError:
        return {
            "epoch": epoch, "betti_0": None, "betti_1": None,
            "entropy": None, "verdict": "ripser_not_installed",
        }

    vecs = extract_geometry_sample(model, val_loader, n=n_sample,
                                   device=device)

    # Persistent homology up to dimension 1 (β0, β1)
    diagrams = ripser(vecs, maxdim=1)['dgms']

    # β0: connected components (infinite persistence = never merged)
    b0 = len([p for p in diagrams[0] if np.isinf(p[1])])

    # β1: loops with sufficient persistence (noise-filtered)
    b1 = len([p for p in diagrams[1]
               if not np.isinf(p[1]) and (p[1] - p[0]) > persistence_threshold])

    # Persistence entropy: Shannon entropy of loop lifetimes
    # Higher = richer topological structure
    lifetimes = np.array([p[1] - p[0] for p in diagrams[1]
                          if not np.isinf(p[1]) and p[1] > p[0]])
    if len(lifetimes) > 0 and lifetimes.sum() > 0:
        p = lifetimes / lifetimes.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-8)))
    else:
        entropy = 0.0

    # Verdict
    if b0 > 20:
        verdict = "flat"        # still disconnected — geometry not forming
    elif b1 == 0 and epoch >= 6:
        verdict = "collapsing"  # forming but no loops — linear manifold risk
    elif b1 > 0:
        verdict = "healthy"     # loops present — structure forming
    else:
        verdict = "forming"     # too early to tell, but not flat

    return {
        "epoch":   epoch,
        "betti_0": b0,
        "betti_1": b1,
        "entropy": entropy,
        "verdict": verdict,
    }


def should_abort_early(trajectory: list) -> tuple:
    """
    Given topology measurements across epochs, decide whether to abort.

    Returns (abort: bool, reason: str).
    Designed to be called after every topological check during training.
    """
    if len(trajectory) < 2:
        return False, "insufficient data"

    latest = trajectory[-1]

    # Hard abort: completely flat at epoch 5+
    if latest["epoch"] >= 5 and latest["betti_0"] is not None:
        if latest["betti_0"] > 20:
            return True, (f"geometry flat at epoch {latest['epoch']} "
                          f"(β0={latest['betti_0']} — not converging)")

    # Soft abort: entropy monotonically decreasing across last 3 checks
    if len(trajectory) >= 3:
        entropies = [t["entropy"] for t in trajectory[-3:]
                     if t["entropy"] is not None]
        if (len(entropies) == 3
                and entropies[2] < entropies[1] < entropies[0]
                and entropies[2] < 0.1):
            return True, (f"persistence entropy collapsing: "
                          f"{entropies[0]:.3f}→{entropies[1]:.3f}→{entropies[2]:.3f}")

    # Good signal: β1 > 0 is strong evidence against abort
    if latest["betti_1"] is not None and latest["betti_1"] > 0:
        return False, f"β1={latest['betti_1']} loops detected — structure forming"

    return False, "no abort signal"


def format_topology_line(topo: dict) -> str:
    """One-line summary for training loop output."""
    if topo.get("verdict") == "ripser_not_installed":
        return "  Topology: ripser not installed — skipping"
    b0 = topo.get("betti_0", "?")
    b1 = topo.get("betti_1", "?")
    h  = topo.get("entropy")
    h_str = f"{h:.3f}" if h is not None else "?"
    verdict = topo.get("verdict", "?")
    return (f"  Topology: β0={b0}  β1={b1}  H={h_str}  [{verdict}]")
