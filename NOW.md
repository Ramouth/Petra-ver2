# NOW.md — Before the Next HPC Run
*Priority actions. Everything here has direct impact on the doover result.*
*Written post-review, post-consulting, post-math-review.*

---

## 1. Topological Early Abort (3-epoch canary)

### Why now

The current canary (separation gap > 0.005 at epoch 3) is a scalar check.
Two models can have identical separation gaps at epoch 3 and completely
different topological trajectories — one develops structure, one collapses
to a near-linear manifold. The separation gap cannot distinguish them.

Topology gives you a *different kind of signal*, not a better version of
the same signal. This is the contribution of both the consultant and the
mathematician converging on the same point.

Cost: ~30 seconds per check, runs every 2 epochs during training.
Requires: `pip install ripser` on HPC.

### What to build: `topology_monitor.py`

```python
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
```

### Integration into train.py

Add after the epoch loop, before checkpointing:

```python
# In train.py — add to imports:
from topology_monitor import (topological_health_check,
                               should_abort_early,
                               format_topology_line)

# In train() function — add before the epoch loop:
topo_trajectory = []
topo_check_every = 2  # check every 2 epochs — lightweight enough

# Inside the epoch loop, after val metrics:
if epoch % topo_check_every == 1:  # epochs 1, 3, 5, ...
    topo = topological_health_check(model, val_loader, epoch,
                                    device=device)
    topo_trajectory.append(topo)
    print(format_topology_line(topo))

    abort, reason = should_abort_early(topo_trajectory)
    if abort:
        print(f"\nTopological early abort at epoch {epoch}: {reason}")
        print(f"  Try a different seed or learning rate.")
        sys.exit(3)  # 3 = topological abort (distinct from other exits)
```

### Integration into select_geometry.py

Add topology score as a secondary selection criterion after separation gap:

```python
# In select_geometry.py — add topology to probe output parsing:
# Parse betti_1 from topology_monitor output in probe logs
# Use as tiebreaker: prefer models with β1 > 0
```

---

## 2. Alignment and Uniformity Metrics (replaces separation gap)

### Why now

Wang & Isola (2020) show that good geometry on a hypersphere requires
*two* properties, not one:

- **Alignment**: same-class vectors should be close (what separation gap measures)
- **Uniformity**: vectors should be spread across the sphere (what separation gap ignores)

A model can have high alignment but low uniformity — winning positions
cluster tightly together but so do losing positions, and they're in the
same region of the sphere. The separation gap would look good. The
geometry would be useless.

Uniformity is cheap to compute and directly measurable. Adding it costs
nothing and improves what you're measuring.

### The metrics

```python
def alignment(win_vecs: np.ndarray, loss_vecs: np.ndarray,
              alpha: float = 2.0) -> float:
    """
    Alignment loss: mean pairwise distance within class.
    Lower = better aligned (same-class vectors are close).

    alpha: exponent (default 2 from Wang & Isola)
    """
    def pairwise_alignment(vecs):
        # Sample pairs for efficiency
        rng = np.random.default_rng(42)
        n = len(vecs)
        n_pairs = min(500, n * (n-1) // 2)
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

    t: bandwidth (default 2 from Wang & Isola)
    """
    rng = np.random.default_rng(42)
    n = len(vecs)
    n_pairs = min(1000, n * (n-1) // 2)
    i_idx = rng.integers(0, n, n_pairs)
    j_idx = rng.integers(0, n, n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    sq_dists = np.sum((vecs[i_idx] - vecs[j_idx]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-t * sq_dists))) + 1e-8)
```

### Where it goes

Add to `probe_geometry.py` Check 2, alongside separation gap:

```python
# In check_label_separation():
align = alignment(win_vecs, loss_vecs)
unif  = uniformity(np.vstack([win_vecs, loss_vecs]))

print(f"\n  Alignment  : {align:.4f}  (lower = tighter same-class clusters)")
print(f"  Uniformity : {unif:.4f}  (lower = better spread across sphere)")
print(f"  Note: both should improve together — one without the other is a warning sign")
```

Add to `select_geometry.py` selection criterion:

```python
# Combined score for A/B selection:
# geometry_score = separation_gap - 0.1 * alignment + 0.1 * (-uniformity)
# (weights are heuristic — adjust after first run)
```

---

## 3. Install ripser on HPC

Before the run:

```bash
pip install ripser --user
# or in your HPC job script:
pip install ripser persim --break-system-packages
```

Verify:
```python
from ripser import ripser
import numpy as np
diagrams = ripser(np.random.randn(100, 128), maxdim=1)
print("ripser working:", len(diagrams['dgms']))
```

---

## Summary: What Changes Before Next HPC Run

| Change | File | Cost | Impact |
|--------|------|------|--------|
| topology_monitor.py | new file | medium | early abort, trajectory signal |
| Topology in train.py | 15 lines | low | wires the abort |
| Alignment + uniformity | probe_geometry.py | 20 lines | better selection metric |
| Topology in select_geometry.py | 10 lines | low | better A/B criterion |
| Install ripser | HPC setup | 1 min | prerequisite |

---

## What Does NOT Change Before Next HPC Run

- vMF contrastive loss — real, but adds hyperparameter before baseline established
- Fréchet mean correction — error is small at 128 dims with well-separated clusters
- Fisher discriminant — separation gap is adequate proxy for now
- Permutation tests — "GOOD/WEAK" sufficient for operational decisions
- Position type balancing — important but belongs after first clean geometry

*These go in LATER.md.*

---

*Co-authored with Claude Sonnet 4.6*
