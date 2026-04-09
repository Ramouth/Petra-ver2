# LATER.md — After First Clean Geometry Result
*These are real and important. They belong after you've established*
*that the geometry works at all — not before.*
*Written post-review, post-consulting, post-math-review.*

---

## The Principle

You need to establish that the geometry works before you optimise
how you measure and train it. Doing the mathematics correctly on a
geometry that doesn't exist yet is expensive and uninformative.

The items in this file become relevant in this order:

```
1. First clean geometry (doover passes test_geometry 5/5)
2. Alignment + uniformity healthy (from NOW.md)
3. β1 > 0 in topology (from NOW.md)
      ↓
   Items in this file become relevant
      ↓
4. GPU run
5. Gaussa
```

---

## 1. vMF Contrastive Loss

### What it is

The von Mises-Fisher contrastive loss gives the geometry a direct
mathematical objective. Currently the geometry is a byproduct of
value prediction — it emerges if conditions are right. With vMF
contrastive loss, the geometry has its own loss term that directly
rewards structuring the hypersphere correctly.

The vMF distribution is the natural analogue of the Gaussian on S^n.
Its concentration parameter κ controls how tightly clustered same-class
vectors should be — analogous to temperature in standard contrastive loss
but with proper spherical geometry grounding.

### Why not now

Adds a hyperparameter (κ) and a loss weight before you've established
what "good geometry" looks like on this architecture. First run tells
you what the geometry probe numbers look like when the architecture is
clean. Then you add the contrastive term and measure the delta.

Adding it now means you can't distinguish "geometry improved because
of board flip" from "geometry improved because of contrastive loss."
Keep variables separate.

### Reference

Davidson et al. — "Hyperspherical Variational Auto-Encoders" (2018, UAI)
arXiv:1804.00891. Section 2 for the vMF density.

Wang & Liu — "Understanding the Behaviour of Contrastive Loss" (2021, CVPR)
arXiv:2012.09740. What contrastive loss actually optimises geometrically.

### Implementation sketch

```python
def vmf_contrastive_loss(geometry_vecs: torch.Tensor,
                          values: torch.Tensor,
                          temperature: float = 0.1,
                          value_threshold: float = 0.3) -> torch.Tensor:
    """
    Von Mises-Fisher contrastive loss on the unit hypersphere.

    geometry_vecs: (B, 128) — already L2-normalised from bottleneck
    values: (B,) — STM-relative, used to derive same/different class
    temperature: κ concentration parameter
    value_threshold: positions within this value distance are "same class"
    """
    # Pairwise cosine similarities (correct on unit sphere)
    sim = geometry_vecs @ geometry_vecs.T / temperature  # (B, B)

    # Soft same-class mask from value proximity
    val_diff = torch.cdist(values.unsqueeze(1), values.unsqueeze(1)).squeeze()
    same = (val_diff < value_threshold).float()

    # Remove diagonal (self-similarity)
    mask = 1 - torch.eye(len(values), device=geometry_vecs.device)
    same = same * mask

    # vMF contrastive: maximise similarity within class
    loss = -torch.log(
        (torch.exp(sim) * same).sum(dim=1) /
        ((torch.exp(sim) * mask).sum(dim=1) + 1e-8)
    ).mean()

    return loss

# In train.py loss computation:
# loss = vloss + policy_weight * ploss + vmf_weight * vmf_loss
# Start vmf_weight at 0.1 — small enough not to dominate value learning
```

### When to add

After first clean geometry result. Sweep κ in {0.05, 0.1, 0.2}.
The alignment metric (from NOW.md) is your signal that it's working.

---

## 2. Fréchet Mean Correction

### What it is

The arithmetic mean of vectors on S^n doesn't lie on S^n. When you
compute win/loss centroids by averaging geometry vectors, the result
is not on the unit sphere. You re-normalise it, which introduces
a systematic bias — the centroid is pulled toward the densest region
of the sphere, not the true geometric centre of the class.

The Fréchet mean is the correct generalisation — the point on the
sphere that minimises the sum of squared geodesic distances to all
class members. It's computed iteratively via the logarithmic map.

### Why not now

The error introduced by the arithmetic mean approximation is small
when clusters are well-separated and points are concentrated. At 128
dimensions with a model that's learning, the bias is real but probably
not the bottleneck. Fix the architecture first, measure the geometry,
then correct the measurement.

The correction matters most when clusters are tight and the centroid
is used for fine-grained decisions — which is exactly what happens
in the A/B selection and the geometry MCTS (the future direction).
Fix it before those, not now.

### Reference

Pennec, X. — "Intrinsic Statistics on Riemannian Manifolds" (2006, JMIV)
Section 3. The logarithmic map and iterative Fréchet mean algorithm.
Free on his website.

### Implementation

```python
def spherical_frechet_mean(vectors: np.ndarray,
                            n_iter: int = 10,
                            tol: float = 1e-6) -> np.ndarray:
    """
    Fréchet mean on S^n via Riemannian gradient descent.

    Replaces arithmetic mean + renormalise in win_loss_centroids()
    and check_label_separation().
    """
    mu = vectors[0] / np.linalg.norm(vectors[0])

    for _ in range(n_iter):
        # Logarithmic map: project all vectors into tangent space at mu
        logs = []
        for v in vectors:
            cos = float(np.clip(np.dot(mu, v), -1.0, 1.0))
            theta = np.arccos(cos)
            if theta < 1e-8:
                logs.append(np.zeros_like(mu))
            else:
                logs.append(theta / np.sin(theta) * (v - cos * mu))

        # Gradient step: move mu toward mean of tangent vectors
        grad = np.mean(logs, axis=0)
        if np.linalg.norm(grad) < tol:
            break
        mu = mu + grad
        mu = mu / np.linalg.norm(mu)  # project back to sphere

    return mu
```

### Where it goes

Replaces `np.mean(..., axis=0)` in:
- `test_geometry.py`: `win_loss_centroids()`
- `probe_geometry.py`: `check_label_separation()`
- `evaluate.py`: `make_geometry_value_fn()`

---

## 3. Fisher Discriminant in Tangent Space

### What it is

The Fisher discriminant ratio is a principled measure of class
separability that normalises by within-class variance. The separation
gap (within-class cosine minus between-class cosine) doesn't normalise
— a large gap with large within-class spread is weaker evidence than
a small gap with tight clusters.

On the hypersphere, the Fisher discriminant is computed in the tangent
space at the grand mean — the linear approximation of the sphere at
that point. This is mathematically correct and computationally simple.

### Why not now

The separation gap is a reasonable proxy. It's correlated with the
Fisher ratio and doesn't require the Fréchet mean (which you're
also deferring). Fix them together — Fréchet mean first, then
Fisher discriminant, since the Fisher discriminant in tangent space
depends on having the correct Fréchet mean as the base point.

### Reference

Harandi et al. — "From Manifold to Manifold" (2014, ECCV)
Section 3. Fisher discriminant on Riemannian manifolds.

### Implementation sketch

```python
def fisher_discriminant_tangent(win_vecs: np.ndarray,
                                 loss_vecs: np.ndarray) -> float:
    """
    Fisher discriminant ratio in tangent space at the grand Fréchet mean.
    Higher = better class separability relative to within-class variance.
    """
    all_vecs = np.vstack([win_vecs, loss_vecs])
    grand_mean = spherical_frechet_mean(all_vecs)  # requires Fréchet mean

    def log_map(mu, v):
        cos = float(np.clip(np.dot(mu, v), -1, 1))
        theta = np.arccos(cos)
        if theta < 1e-8:
            return np.zeros_like(mu)
        return theta / np.sin(theta) * (v - cos * mu)

    # Project to tangent space
    win_t  = np.array([log_map(grand_mean, v) for v in win_vecs])
    loss_t = np.array([log_map(grand_mean, v) for v in loss_vecs])

    mu_win  = win_t.mean(axis=0)
    mu_loss = loss_t.mean(axis=0)

    # Between-class scatter
    between = float(np.linalg.norm(mu_win - mu_loss) ** 2)

    # Within-class scatter
    within_win  = float(np.mean([np.linalg.norm(v - mu_win)  ** 2 for v in win_t]))
    within_loss = float(np.mean([np.linalg.norm(v - mu_loss) ** 2 for v in loss_t]))
    within = (within_win + within_loss) / 2

    return between / (within + 1e-8)
```

---

## 4. Permutation Tests for Significance

### What it is

The nearest-neighbour consistency check (Check 4) currently reports
"GOOD/WEAK/NONE" based on lift score thresholds. These thresholds are
heuristic — there's no mathematical basis for saying lift > 0.1 is
"GOOD."

A permutation test gives you a p-value: the probability of observing
the measured lift under the null hypothesis that geometry has no
structure. This is the correct way to make the claim "the geometry is
real" rather than "the geometry looks good to me."

### Why not now

For operational decisions (should I run the gate? should I proceed to
GPU?) the heuristic thresholds are sufficient. The p-value matters when
you want to make a claim that the geometry is real to someone else —
a paper, a collaborator, yourself with rigour. That moment comes after
the GPU run establishes results worth defending.

### Reference

Good, P. — *Permutation, Parametric and Bootstrap Tests* (2005) Ch. 1, 3.

### Implementation sketch

```python
def nn_consistency_permutation_test(vecs: np.ndarray,
                                     values: np.ndarray,
                                     k: int = 5,
                                     n_probe: int = 200,
                                     n_permutations: int = 1000) -> dict:
    """Returns observed lift, null distribution, and p-value."""
    from probe_geometry import _label_class, compute_nn_match_rate

    labels = np.array([_label_class(v) for v in values])
    observed = compute_nn_match_rate(vecs, labels, k, n_probe)

    rng = np.random.default_rng(42)
    null_rates = [
        compute_nn_match_rate(vecs, rng.permutation(labels), k, n_probe)
        for _ in range(n_permutations)
    ]
    null = np.array(null_rates)
    p_value = float((null >= observed).mean())

    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_std":  float(null.std()),
        "p_value":   p_value,
        "significant": p_value < 0.05,
    }
```

---

## 5. Position Type Balancing

### What it is

Self-play positions from a weak model cluster in specific regions of
position space — typically decisive late-game positions. The geometry
learns those well and generalises poorly to the full range of chess
positions. This is a data distribution problem that architecture fixes
can't see.

The fix: tag each position by structural type (material advantage,
king safety, pawn structure, endgame) and sample training batches
balanced across types. Force the model to see the full range equally.

### Why not now

You don't know yet whether data distribution is the bottleneck. It
might be architecture. Establish clean geometry on the current data
distribution first. If geometry health plateaus despite architecture
being correct, data distribution is the next lever.

Also: position tagging requires defining the types, which requires
chess knowledge that should be grounded in what the geometry actually
learns — not imposed before training. Let the geometry show you what
it separates, then reinforce those separations.

### Implementation direction

```python
def tag_position(board: chess.Board) -> str:
    """Structural type for balanced sampling."""
    n_pieces = len(board.piece_map())
    if n_pieces <= 10:
        return "endgame"

    material_balance = sum(
        PIECE_VALUE[p.piece_type] * (1 if p.color == chess.WHITE else -1)
        for p in board.piece_map().values()
        if p.piece_type != chess.KING
    )
    if abs(material_balance) >= 3:
        return "material_advantage"

    # King safety proxy: pawns in front of castled king
    # Pawn structure: isolated / passed / doubled pawns
    # ... (expand based on what geometry separates)
    return "complex"
```

---

## 6. Structural Analogy Test

*Listed in FIX.md as "the missing test." Belongs here in terms of timing.*

### Why not now

Can't implement this meaningfully until the geometry exists to test.
The test requires knowing which structural features the geometry has
actually learned — you discover those from the first GPU run, then
design positions that isolate them.

What you *can* do now: design the position pairs in advance, based on
chess knowledge, so the test is ready to run immediately after the GPU
run produces a model worth testing.

```python
# Pre-designed position pairs for structural analogy test
# (run these immediately after GPU training)

STRUCTURAL_PAIRS = [
    # Rook endgame — opposite values, same structure
    ("rook_endgame_winning", chess.Board("8/8/8/8/4k3/8/4R3/4K3 w - - 0 1")),
    ("rook_endgame_losing",  chess.Board("8/8/8/8/4K3/8/4r3/4k3 w - - 0 1")),

    # Open vs closed — same material, different structure
    ("open_position",   chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1")),
    ("closed_position", chess.Board("r1bqk2r/pp2bppp/2np1n2/2p1p3/2P1P3/2NP1N2/PP2BPPP/R1BQK2R w KQkq - 0 1")),

    # Pawn endgame — same type, different winner
    ("pawn_endgame_winning", chess.Board("8/4k3/8/4P3/8/8/8/4K3 w - - 0 1")),
    ("pawn_endgame_losing",  chess.Board("8/4K3/8/4p3/8/8/8/4k3 w - - 0 1")),
]

# Test: intra-type cosine > inter-type cosine
# (after board flip: winning and losing versions should cluster
#  together if geometry encodes structure, not just value)
```

---

## Reading Order (for reference)

When you're ready for this file's content:

```
Start here:
  Wang & Isola (2020) — alignment and uniformity (already in NOW.md)
  Ghrist (2008)       — topology intuition (already reading)

Then:
  Pennec (2006)       — before implementing Fréchet mean
  Davidson (2018)     — before implementing vMF contrastive loss
  Harandi (2014)      — before implementing Fisher discriminant

For Gaussa specifically:
  Carlsson (2009)     — topology and data, the bridge
  Lum et al. (2013)   — Mapper algorithm, knowledge graphs
```

Full reading list in REFERENCES.md (from the mathematician's session).

---

*Co-authored with Claude Sonnet 4.6*
