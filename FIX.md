# FIX.md — Petra Doover: Code Review & Fix Plan
*Written after full code review session, April 2026.*
*Context: post-R6, pre-GPU. Full retrain from scratch.*

---

## Why a Doover

The architecture has been patching a problem that originates at the input layer.
The progression tells the story:

- **R4**: ReLU bottleneck → 26/128 dead dimensions, geometry stagnated, ELO gains
  came from deeper search not geometry
- **R6**: Tanh bottleneck → fixed dead dimensions, but b/w asymmetry persists
  (3/5 STM symmetry test), Black-up-queen sanity check fails
- **Root cause**: `board_to_tensor` encodes from White's absolute perspective.
  The CNN must learn two spatial grammars — one for White to move, one for Black.
  The geometry can't unify them through 128 dimensions.

The policy head solved this by brute-forcing both grammars with 4096 output dimensions.
The geometry bottleneck couldn't — 128 dimensions isn't enough room to encode
two grammars *and* the strategic content.

The downstream patches (STM sign multiplication, antipodal loss, STM-aware curriculum)
were all correcting a problem that should be fixed at the source.

**Two changes fix the root cause:**
1. Board flipping in `board_to_tensor` — one spatial grammar for both sides
2. L2 normalisation in the bottleneck — geometry metric matches how we measure it

Everything else in the codebase is sound. This is a targeted doover, not a rewrite.

---

## Architecture Changes

### Change 1: Board flip in `board_to_tensor` (board.py)

**What**: When Black is to move, rotate the board 180° so the CNN always sees
"my pieces at the bottom." Planes 0–5 always encode the mover's pieces,
planes 6–11 always encode the opponent's pieces.

**Why**: The CNN learns one spatial grammar. A passed pawn on rank 6 looks
the same regardless of whose pawn it is. King safety patterns are unified.
The geometry bottleneck can encode strategic concepts without spending
dimensions on the colour convention.

**Evidence**: Policy learned both grammars (4096 output dims, enough room).
Geometry didn't (128 dims, not enough room for two grammars + strategy).
3/5 on STM symmetry test confirms the input encoding is the constraint.

**Consequence**: `move_to_index` must also flip when Black is to move.
This cascades to `data.py`, `selfplay.py`, and `model.py`.

### Change 2: L2 normalisation bottleneck (model.py)

**What**: Replace `Tanh` in the bottleneck with L2 normalisation (unit hypersphere).

**Why**: `probe_geometry.py` measures everything with cosine similarity.
Tanh doesn't produce vectors where cosine is the natural metric — the geometry
is trained with one metric and measured with another. L2 normalisation fixes this:
every vector lives on the unit sphere, cosine similarity is the correct distance
by construction.

**Additional benefit**: No saturation, gradients flow freely through the bottleneck
to the ResBlocks. The 128 dimensions are used more uniformly.

**Evidence from literature**: FaceNet, SimCLR, and metric learning systems
universally use L2 normalisation at the embedding layer for this reason.

### Change 3: Value head ReLU → Tanh (model.py)

**What**: Replace `ReLU` in the value head with `Tanh`.

**Why**: R6 status note identified this directly:
> "ReLU breaks antisymmetry at the head level — if the bottleneck correctly
> flips sign for color-symmetric positions, the ReLU in the value head kills
> the negative component."

After board flipping, the geometry vector for a losing position will have
negative components. ReLU kills them. Tanh preserves them.

### Change 4: Thin value head (model.py)

**What**: Remove the hidden layer from the value head.
`Linear(128→64) + Tanh + Linear(64→1) + Tanh` → `Linear(128→1) + Tanh`

**Why**: With a rich 128-dim geometry on the unit sphere, the value head just
needs to find the right direction. A deep value head compensates for a weak
geometry — that's not what we want. The geometry should do the work; the head
should read it.

---

## File-by-File Fix Plan

### board.py — Rewrite

**Problems:**
- Board always encoded from White's absolute perspective
- `move_to_index` returns absolute coordinates regardless of turn
- `PIECE_TO_PLANE` concept changes after flip

**Fix — new encoding convention:**
```
Planes 0–5:   STM's pieces      (P N B R Q K)
Planes 6–11:  Opponent's pieces (P N B R Q K)
Plane 12:     Always 1.0 (STM convention — kept for compat)
Plane 13:     Castling rights (STM-relative: rank 0 = STM back rank)

When Black to move: flip ranks (rank → 7 - rank)
move_to_index(move, flip=False) — add flip parameter
```

**Helper needed:**
```python
def _flip_square(sq: int) -> int:
    rank = sq >> 3
    file = sq & 7
    return ((7 - rank) << 3) | file
```

**Call sites that need updating** (move_to_index gains `flip` parameter):
- `data.py` line 201 — inlined move encoding
- `selfplay.py` lines 234, 278 — move_idxs and visit_dist encoding
- `model.py` lines 115, 118 — policy() method

---

### model.py — Targeted changes

**Problems:**
1. Tanh bottleneck → mismatched metric (L2 fixes this)
2. ReLU in value head → kills negative components
3. Deep value head → compensates for weak geometry
4. `policy()` doesn't pass flip to `move_to_index`
5. STM sign multiplication in `_piece_geometry` — remove (no longer needed
   after board flip; would double-correct and break everything)

**Fixes:**
```python
# 1. L2 normalisation
class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=-1)

self.bottleneck = nn.Sequential(
    nn.Flatten(),
    nn.Linear(channels * 8 * 8, bottleneck_dim),
    L2Norm(),
)

# 2 + 3. Thin value head with Tanh throughout
self.value_head = nn.Sequential(
    nn.Linear(bottleneck_dim, 1),
    nn.Tanh(),
)

# 4. Flip in policy()
flip = (board.turn == chess.BLACK)
for move in board.legal_moves:
    idx = move_to_index(move, flip=flip)
    mask[idx] = logits[idx]
probs = torch.softmax(mask, dim=0)
return {move: probs[move_to_index(move, flip=flip)].item()
        for move in board.legal_moves}
```

**Critical**: Search for any STM sign multiplication remaining in model.py
and remove it. The board flip handles this at the input — any downstream
sign correction would double-correct.

---

### data.py — One fix

**Problem**: Line 201 inlines move encoding without flip:
```python
move_idx_buf[count] = move.from_square * 64 + move.to_square
```

**Fix:**
```python
from board import move_to_index  # add to imports
move_idx_buf[count] = move_to_index(move, flip=(b.turn == chess.BLACK))
```

---

### selfplay.py — Four fixes

**Fix 1**: Add `board.turn` to positions tuple (line 135):
```python
positions.append((
    board.fen(),
    board_to_tensor(board).numpy(),
    visit_dist,
    half_move,
    board.turn,   # ← add this
))
```

**Fix 2**: Use turn for flip in `_collect` (line 228):
```python
for fen, tensor, visit_dist, _, turn in result["positions"]:
    flip = (turn == chess.BLACK)
    ...
    all_move_idxs.append(move_to_index(best_move, flip=flip) if best_move else 0)
    all_turns.append(turn)   # parallel list for fix 3
```

**Fix 3**: Use turn for visit distribution encoding (line 278):
```python
for i, vd in enumerate(all_visit_dists):
    flip = (all_turns[i] == chess.BLACK)
    for move, prob in vd.items():
        visit_dist_t[i, move_to_index(move, flip=flip)] = prob
```

**Fix 4**: Move resign check *after* recording position (line 122).
Currently the last `resign_consecutive` positions before resignation are
never recorded — these are the most decisive, highest-signal positions:
```python
# Record FIRST
positions.append((board.fen(), board_to_tensor(board).numpy(),
                  visit_dist, half_move, board.turn))
board.push(move)

# Then resign check (on the new position)
val = model.value(board, device)
...
```

**Note on MAX_POSITIONS_PER_GAME**: Currently 12. R6 status note confirms
this produced only 5,782 positions — "very small for a cold start." ZIGZAG.md
spec says 50. Confirm before changing, but 50 is almost certainly correct.

---

### train.py — Two fixes

**Fix 1**: Memory efficiency in `mix_anchor` (line 104).
Currently allocates one-hot for entire anchor dataset before sampling.
At 200k positions that's ~3GB temporary:
```python
# Sample first, then build one-hot
idx = torch.randperm(n_anchor)[:n_sample]
a_sampled_idxs = a["move_idxs"][idx]
vd = torch.zeros(n_sample, 4096, dtype=torch.float32)
vd[torch.arange(n_sample), a_sampled_idxs] = 1.0
```

**Fix 2**: LR scheduler / early stopping patience mismatch.
LR halves after 3 epochs no improvement, early stop after 5 —
barely one reduction before stopping. Either tighten scheduler to 2
or increase early stop patience to 8-10:
```python
# Option A: tighter scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=2, factor=0.5, min_lr=1e-5
)
# Option B: more patient early stopping (recommended for cold start)
patience: int = 10
```

---

### test_geometry.py — Two fixes

**Fix 1**: Test 3 tests the wrong thing.
Current code expects antipodal (cos < -0.3). After board flipping,
`board.mirror()` (swap colors + STM) should produce *similar* geometry
because both positions look identical to the CNN after flipping.
Expected outcome is correlation, not antipodal:
```python
# New threshold:
ok = cos > 0.9   # similar, not antipodal
status = PASS if cos > 0.85 else (WARN if cos > 0.7 else FAIL)
```

**Fix 2**: `win_loss_centroids` uses only White-to-move positions.
After board flipping both should look the same to the CNN, but including
Black-to-move positions makes the centroid more robust and tests the fix:
```python
win_positions = [
    chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1"),  # white to move, white winning
    chess.Board("4K3/8/8/8/8/8/8/4k2q b - - 0 1"),  # black to move, black winning
    chess.Board("4k3/8/8/2Q5/8/8/8/4K3 w - - 0 1"),
    chess.Board("4K3/8/8/2q5/8/8/8/4k3 b - - 0 1"),
]
# Mirror for loss_positions
```

---

### probe_geometry.py — Three fixes

**Fix 1**: Remove White-absolute perspective conversion (lines 50-52).
After board flipping, geometry vectors are STM-relative. Converting values
to White-absolute would mislabel Black-to-move winning positions as losses:
```python
# Remove:
# white_to_move = tensors[:, 12, 0, 0].numpy() > 0.5
# values = np.where(white_to_move, values, -values)
# Keep values STM-relative as stored
```

**Fix 2**: Replace mean norm with effective rank after L2 (line 107).
After L2 normalisation all norms are 1.0 — uninformative:
```python
effective_rank = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
print(f"  Effective rank: {effective_rank:.1f} / 128")
# Healthy: > 30. Collapsed: < 10.
```

**Fix 3**: Extend `_label_class` to bucket continuous SF values.
After zigzag with SF re-labeling, most values are continuous not discrete.
Current code returns "invalid" for anything that isn't exactly ±1 or ±0.1,
silently depressing Check 4 lift scores:
```python
def _label_class(v: float) -> str:
    if v > 0.5:  return "win"
    if v < -0.5: return "loss"
    return "draw"
```

---

### evaluate.py — Two fixes

**Fix 1**: Gate exit code not wired. `zigzag.py` uses `rc == 0` as the gate
check but `evaluate.py` always exits 0. Add to end of `main()`:
```python
if 5 in results and results[5]["win_rate"] <= 0.55:
    sys.exit(1)   # signals zigzag.py to stop
```

**Fix 2**: Geometry anchors are all White-to-move. Same fix as test_geometry.py.
Add Black-to-move equivalents to `_GEOMETRY_ANCHORS_WIN` and `_GEOMETRY_ANCHORS_LOSS`.
After board flipping, these should produce identical geometry vectors to their
White-to-move equivalents — making the anchor set more robust and testing the fix.

---

### zigzag.py — Two improvements

**Improvement 1**: Add geometry probe to the loop automatically.
Currently you have to run probe_geometry.py manually between rounds:
```python
# After Step 3 train, before Step 4 gate:
print(f"\n--- Step 3b: Geometry probe ---")
_run([
    sys.executable, os.path.join(SRC_DIR, "probe_geometry.py"),
    "--model",   os.path.join(out_dir, "best.pt"),
    "--dataset", sf_pt,
], dry_run)
```

**Improvement 2**: Wire up `_check_gate` or remove it.
Currently dead code — `evaluate.py` exit code fix (above) makes `rc == 0`
reliable, so `_check_gate` log parsing is no longer needed. Remove it.

---

## What Doesn't Change

- `mcts.py` — correct as-is, no changes needed
- Zigzag curriculum (round parameters, SF depths, game counts)
- Training loop structure (MSE value loss + KL policy loss)
- Self-play parallel worker architecture
- Opening book and pilot mode
- All HPC job scripts (paths unchanged)

---

## Geometry Test Expected Outcomes After Doover

With board flip + L2 + clean architecture, starting from scratch:

| Metric | R4 | R6 | Expected post-doover |
|--------|-----|-----|----------------------|
| Test 3 STM symmetry | ? | 3/5 | 5/5 |
| Dead dimensions | 26/128 | ~0 | < 5/128 |
| Centroid cosine | 0.869 | ? | < 0.5 |
| Separation gap | 0.048 | ? | > 0.10 |
| NN lift (Check 4) | ? | ? | > 0.10 |
| Black-up-queen sanity | ? | ✗ | ✓ |
| Gate (vs material) | ? | ? | > 55% |

---

## The Missing Test (After Doover)

Current tests measure *value prediction in geometry space* — they all reduce to
"does the geometry predict the outcome correctly." This is necessary but not
sufficient for Gaussa.

What's missing is a **structural analogy test**: do positions with the same
strategic character cluster together *independently of their value*?

Concrete example:
- White up a rook, closed pawn structure → geometry G1
- Black up a rook, closed pawn structure → geometry G2  (opposite value)
- White up a rook, open position        → geometry G3   (same value as G1)

If the geometry encodes structure: G1 and G2 should share dimensions encoding
"rook advantage + closed structure" even though their values are opposite.
G1 and G3 should differ in the "pawn structure" dimensions.

This test is the real gate before Gaussa. It doesn't need to pass before the
doover — but it needs to pass before declaring the geometry ready for transfer.

---

## A/B Geometry Selection

### The problem with single-seed training

A cold start on new architecture has real variance. The random seed affects weight
initialisation and data shuffling. Two runs with identical hyperparameters can
produce meaningfully different geometry — one lands in a basin where the win/loss
axis is clean, another doesn't. With one run you can't distinguish "architecture
doesn't work" from "this seed was unlucky."

Previous rounds never noticed this because they fine-tuned from an existing model
(the geometry was already partially formed). The doover starts from scratch —
variance is at its maximum exactly when you can least afford it.

### The fix: run N seeds, select by geometry quality

Run 2-3 training jobs in parallel with different seeds. Pick the winner by
**geometry probe separation gap** (Check 2), not val loss.

Val loss measures value prediction quality. Geometry quality is what you actually
want. A model with slightly higher val loss but a cleaner geometry is strictly
better for this project. These two things are correlated but not identical —
you want to optimise the thing you actually care about.

```
Round 1 bootstrap:
  seed 0 → models/candidate_s0/best.pt → probe → gap=0.031
  seed 1 → models/candidate_s1/best.pt → probe → gap=0.089  ← winner
  seed 2 → models/candidate_s2/best.pt → probe → gap=0.044

select_geometry.py picks seed 1, copies to models/zigzag/r1/best.pt
```

### What to vary

- **Seed** (0, 1, 2) — variance from weight init and data shuffle. Always vary this.
- **Learning rate** (optional) — e.g. 3e-4, 5e-4, 7e-4. The interaction between
  LR and L2 geometry is untested on this architecture. If seeds alone don't
  produce enough spread, vary LR across candidates.
- **Anchor fraction** (optional) — 0.10, 0.15, 0.20. Controls how much the
  supervised data anchors the geometry vs self-play pulling it.

Do not vary more than one thing at a time on the first run. The goal is to
find a good geometry, not to understand hyperparameter sensitivity yet.

### HPC cost

On a 32-core node: three training jobs run in parallel, finishing in the same
wall time as one. The geometry probe adds ~5 minutes per candidate.
Total overhead for 3-way selection on a 2-3 hour training job: ~5 minutes.

This is essentially free.

### Selection criterion

Primary: **separation gap** from Check 2 (within-class vs between-class cosine distance).
Secondary: **effective rank** from Check 1 (how many dimensions are active).
Tiebreaker: val loss.

If all candidates have gap < 0.02 (flat geometry), something is architecturally
wrong — don't pick the least bad one, investigate before proceeding.

### Implementation

`src/select_geometry.py` — takes a list of candidate model paths, runs
probe_geometry.py on each, parses the separation gap, copies the winner
to the target path. See that file for usage.

### When to use A/B selection

- **Round 1 (cold start)**: always. Variance is highest here.
- **Rounds 2-4 (fine-tuning)**: optional. Variance decreases as the geometry
  stabilises. Use if round 1 geometry was marginal (gap 0.02-0.05).
- **GPU run**: not needed. GPU allows enough compute per run that a single
  seed with longer training dominates any seed selection benefit.

---

## Implementation Order

```
1. board.py          — root fix, everything cascades from here
2. model.py          — L2, thin head, flip in policy(), remove STM sign mult
3. data.py           — one line
4. selfplay.py       — flip fixes, resign ordering, MAX_POSITIONS
5. train.py          — memory fix, patience
6. test_geometry.py  — Test 3, balanced anchors
7. probe_geometry.py — perspective flip removal, effective rank, label bucketing
8. evaluate.py       — gate exit code, balanced anchors
9. zigzag.py         — geometry probe in loop, remove dead gate code

Then: single HPC training run → test_geometry.py → if 5/5 → GPU.
```

---

*Co-authored with Claude Sonnet 4.6*
