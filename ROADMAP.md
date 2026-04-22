# Petra — Geometry-Driven Engine Roadmap

## Current Purpose

Petra is no longer aiming merely to be a stronger value+policy engine with an
interesting latent space. The purpose is now narrower and stricter:

**Build a chess engine whose search can eventually be driven by geometry, not
just a scalar value head.**

That changes what counts as progress:

- Better ELO is necessary but not sufficient
- A clean win/loss axis is necessary but not sufficient
- The bottleneck must develop **distributed, non-collapsed structure**
- The data and training pipeline must be trustworthy under STM-relative encoding
- The current training setup must be treated as **preparation for geometry-driven
  search**, not as the final engine design

## Immediate Goal

Achieve the current doover run cleanly:

1. Parse Lichess into a validated STM-relative dataset
2. Re-label with Stockfish while preserving the original train/val boundary
3. Train on broad SF-Lichess data mixed with 15% endgame anchors
4. Probe geometry with **effective rank as the primary gate**

The current run is successful only if it produces:

- Effective rank `> 30`
- A meaningful but non-collapsed win/loss axis
- Correct sanity behaviour on basic asymmetric endgames
- Validation metrics that are not polluted by train/val leakage

Centroid cosine and separation gap are secondary diagnostics. They can look
excellent in a degenerate 1D geometry, which is not the target.

## What The Current Setup Is For

The current codebase is **not yet a geometry-driven engine**. It is the
foundation stage that must answer a simpler question first:

> Can Petra learn a broad, stable, non-collapsed geometry from chess positions
> that is rich enough to justify making geometry part of search?

If the answer is no, geometry-driven MCTS is premature.
If the answer is yes, the next phase is to make search depend on geometry
directly rather than treating geometry as a byproduct of value training.

## Execution Roadmap

### Milestone A — Trustworthy Geometry Pretraining

Purpose: produce a representation that is worth using in search.

Requirements:
- STM-relative board encoding everywhere
- Strict dataset validation restored under STM-relative assumptions
- SF re-evaluation preserves original train/val split
- Endgame anchors are true antipodal partners
- Anchor mixing preserves anchor policy targets instead of replacing them with
  arbitrary one-hot labels

Status:
- Board flip / STM-relative move indexing: implemented
- L2 bottleneck + thin value head: implemented
- Probe uses effective rank: implemented
- Parser validation repaired for STM-relative turn handling: implemented
- GPU mixed-anchor training job: implemented

### Milestone B — Broad Geometry Gate

Run this exact path:

```bash
bsub < jobs/parse_lichess.sh
bsub < jobs/reeval_sf.sh
bsub < jobs/gen_endgame.sh
bsub < jobs/train_sf_gpu.sh
python3 src/probe_geometry.py --model models/sf_gpu/best.pt --dataset data/endgame_anchor.pt --n 5000
```

Gate:
- Effective rank `> 30`
- No obvious sign failures on asymmetric sanity positions
- Probe shows broad usage of dimensions, not merely a perfect binary axis

Interpretation:
- If rank stays low, the setup is still learning a scalar disguised as geometry
- If rank rises while retaining a clean win/loss axis, the representation stage
  is greenlit

### Milestone C — Make Geometry Part Of Search

Only after Milestone B passes.

Add geometry-dependent search terms:

- Measure move-induced geometry deltas `g(s') - g(s)`
- Test whether winning moves align with a learned directional structure
- Add a geometry bonus to MCTS selection and compare against standard scalar MCTS
- Evaluate whether geometry improves move quality beyond the value head alone

This is the point where Petra becomes a candidate geometry-driven engine rather
than a value engine with a well-monitored bottleneck.

### Milestone D — Geometry-Driven Planning

Longer-term, if geometry search signal is real:

- Learn a transition model `f(g_t, move) -> g_{t+1}`
- Evaluate move trajectories in geometry space
- Use board state for legality and terminal truth, geometry for planning bias

That is the actual engine thesis. Everything before it is infrastructure.

## Current Critical Risks

- The setup can still produce strong scalar value performance without geometry
  becoming action-relevant
- Effective rank can remain low even with good centroids and good sanity checks
- Search currently uses scalar value only; geometry is still diagnostic
- Endgame anchors help orient the space but do not by themselves make search
  geometry-driven

## Historical Archive

The sections below are retained as project history. They describe how Petra
reached the current doover, but they are no longer the authoritative statement
of purpose or gating.

### Terminology

- **Session** — a block of work time (roughly a conversation/day). Sessions are
  numbered continuously across the whole project history.
- **Round** — one training run producing one named model checkpoint
  (configure → train → probe → gate). Rounds reset to 1 at each doover.
- **Doover** — a fundamental restart: new architecture, new data strategy, or
  both. Old model weights are not carried forward. Marked explicitly in session
  headers. Round numbering restarts from 1.

---

## Context

Phase 0 (2026-03-20 to 2026-03-24) failed. The geometry encoder (walk profile → contrastive loss → 128-dim space → value) cannot be trained on correct chess data — contrastive loss collapses because most middlegame positions have near-identical values. The v6 encoder that appeared to work was exploiting a sign-error bug in training data, not learning chess geometry.

Phase 0 end state: Petra = Maia-1500 policy + near-constant value function. Strictly worse than Maia-1500 greedy alone.

**Phase 1 thesis (⟳ DOOVER 1):** Replace the geometry encoder with a direct supervised value+policy network (PetraNet) trained on Lichess game outcomes. Validate with ELO before anything else. Build on what works. *Rounds reset to 1.*

---

## Phase 1 Milestones

### Milestone 1 — PetraNet architecture + MCTS ✅
CNN on raw board tensor (14×8×8). Joint value head (MSE on game outcome) and policy head (cross-entropy on move played). MCTS wired to PetraNet value + policy.

### Milestone 2 — Data pipeline with integrity validation ✅
Lichess PGN parser with 6 integrity checks (label values, sign correctness, label distribution, side-to-move balance, king presence, within-game sign consistency). Game-level train/val split (no leakage). Source: `lichess_db_standard_rated_2025-01.pgn.zst`.

### Milestone 3 — Supervised training loop ✅
Joint value + policy training. Early stopping on val loss (patience=5). Best checkpoint saved separately. Post-training sign sanity checks.

### Milestone 4 — Ablation game runner + evaluation agents ✅
Game runner for head-to-head evaluation. Evaluation agents for ELO testing.

---

## Session 1 Results (2026-03-25)

### Run 1 — Lichess game-outcome labels (150k games)
- **Dataset:** 1.19M positions, 150k games, min Elo 1500. All 6 validation checks passed.
- **Training:** 15 epochs, ~70 min/epoch on CPU. Best val loss: 3.3008.
- **Policy:** Top-1 35.8%, Top-5 70.4% — strong.
- **Value:** R²=0.181 — weak. Game outcome is too noisy a label.
- **Ablation:** MCTS(learned) lost 0-100 to MCTS(material). Gate FAILED.
- **Geometry probe:** 128-dim space healthy (top-1 eigenvalue 12.7%, not collapsed). But win/loss centroids nearly identical (cosine sim 0.9951) — space organised by board structure, not value.

### Conclusion
Policy head is a real asset. Value head needs a stronger training signal. Game outcome labels are insufficient — positions early in a game have weak correlation with the eventual winner.

---

## Session 1 Continued — SF Model (2026-03-25)

### Step 5 — Stockfish re-evaluation ✅
Re-evaluated 200k positions from dataset.pt with Stockfish depth 10. Labels replaced with `tanh(cp/400)`.

- **Label std: 0.565**, 43% decisive, 20% equal
- Saved → `dataset_sf.pt` (190k train / 10k val)

### Step 6 — Retrain on SF labels ✅
- **Best val loss: 4.2461**, R²=0.483, Top-1 21.2%, Top-5 45.0%
- All sanity checks passed. Start position value=-0.138 (slight negative bias, noted)
- Saved → `models/sf/best.pt`

### Step 7 — Ablation on SF model ✅
- Step 2 Greedy: 74% (+182 ELO)
- Step 3 MCTS(zero): 61.5% (+81 ELO)
- Step 4 MCTS(material): 89% (+363 ELO)
- Step 5 MCTS(learned) at n_sim=20: **54%** — gate FAILED (threshold 55%)
- Step 5 repeated at n_sim=100: **54.5%** — consistent positive trend, still below threshold

### Step 8 — Geometry probe on SF model ✅
- Win/loss centroid cosine: **0.9192** (vs 0.9951 for Lichess model — significant improvement)
- Top-1 eigenvalue: 23% (healthy spread)
- NN label consistency: 0.901
- Rated WEAK but meaningful separation emerging

### Decision: proceed to zigzag ✅
54-54.5% is a consistent positive signal. More supervised training on Lichess positions has diminishing returns — the model needs to see positions from its own play. Zigzag chosen.

---

## Session 2 — Zigzag Design + HPC Setup (2026-03-25)

### Fixes
- **evaluate.py**: added `--temp-moves` (default 10) — MCTSAgent was using temperature=0 making all 100 games identical. Fixed by using temperature=1 for first N half-moves.
- **data.py**: uint8 tensor storage, zst support, numpy pre-allocation
- **train.py**: `.float()` conversion for uint8 tensors

### ZIGZAG.md written ✅
Full design document: 4-round curriculum, data strategy, policy loss change, temperature schedule, LR schedule, failure modes, HPC plan, open questions.

### DTU HPC (gbar) setup ✅
- CPU-only access confirmed (no GPU for now)
- `petra-env` venv created, numpy/chess/torch installed
- Stockfish binary installed at `~/bin/stockfish`
- Project cloned to `~/Petra-Phase1`

---

## Session 3 — Zigzag Implementation (2026-03-26)

### Built ✅
- **selfplay.py**: self-play game generation. MCTS with temperature schedule, Dirichlet noise, resign threshold, position sampling. `--workers N` for HPC parallelism.
- **zigzag.py**: orchestration of the full loop (selfplay → reeval → train → gate).
- **train.py**: dense policy loss (KL over visit distributions) for self-play datasets; `--init-model` for fine-tuning from prior round.

### Round 1 prototype (50 games, n_sim=40, SF depth 12) — in progress
- Self-play: 50 games, 600 positions, 22 min ✅
- SF re-label: 600 positions at depth 12, 20s. 79% decisive ✅
- Train (fine-tune from models/sf/best.pt): val R²=0.781, all sanity checks passed ✅
- Gate evaluation (100 games, n_sim=40): **in progress** — 10/100 at 55%

---

## Sessions 4–6 — Zigzag R1–R4, Geometry Analysis (2026-03-27 to 2026-03-29)

### Zigzag round results

| Round | Win rate | Opponent | Notes |
|-------|----------|----------|-------|
| R1 | 54.5% | MCTS(material) | Marginal pass. SF labels, n_sim=40 |
| R2 | 61.0% | MCTS(material) | Strong pass. SF labels, n_sim=80 |
| R3 | 59.0% (45% vs R2) | MCTS(material) | Regression vs R2. Root cause: same data distribution, early stopping at epoch 1 |
| R4 | 67.0% | MCTS(material) | Strong. 200 games, n_sim=400, init from R2. Deeper search hypothesis confirmed |

R4 change: reset to R2 weights + deeper MCTS. Gate improved significantly. The gain is from search depth, not from geometry improvement.

### PGN analysis (R3 and R4 gates)
- 25–29% of games hit the 300-move limit — not bare king shuffles but complex endgames
- Type A (5/25): queen vs bare king, model can't deliver checkmate (geometry failure)
- Type B (20/25): R+P vs R+P, Q+B vs Q+B — some theoretical draws, some conversion failures
- Conclusion: Petra plays real chess. The failure mode is endgame conversion, not structural chaos.

### Geometry probe — critical finding (2026-03-29)

Ran `compare_geometry.py` across R1, R2, R4 using `selfplay_r1_full_sf.pt` as fixed reference.

| Metric | R1 | R2 | R4 |
|--------|----|----|-----|
| Top-1 eigenvalue (%) | 21.5 | 21.7 | 21.5 |
| Centroid cosine sim | 0.883 | 0.871 | 0.869 |
| Separation gap | 0.050 | 0.057 | 0.048 |
| NN consistency | 0.912 | 0.903 | 0.902 |
| Mean vec norm | 92.4 | 90.2 | 87.8 |
| KQ vs K (White) | ✓ | ✓ | ✗ FAIL |
| K vs KQ (Black) | ✗ FAIL | ✗ FAIL | ✗ FAIL |

**The encoder is frozen.** SF labels do not improve geometry across rounds. ELO gains come from deeper MCTS search, not better representation. K vs KQ (White bare king = losing) has never been classified correctly in any round — a systemic bias.

### Architectural root cause: ReLU in the bottleneck

The bottleneck is `Linear(4096→128) + ReLU`. ReLU forces all geometry values ≥ 0. Consequence:
- 57% of geometry values exactly zero
- 26 of 128 dimensions permanently dead (never activate on any position)
- Only ~27 dimensions distinguish win from loss
- Win centroid mean activation (2.80) ≈ loss centroid mean activation (2.83)
- Win/loss centroids cannot be antipodal — both live in the positive orthant

The value head compensates by memorising SF outputs on top of a crippled representation. The geometry hypothesis has never been properly testable with this architecture.

**Fix:** Replace `ReLU` with `Tanh` in the bottleneck. Done in this session (model.py). Tanh allows negative values → win/loss can occupy opposite sides of the origin → all 128 dimensions become usable.

---

## R5–R7 Plan (2026-03-29)

### R5 — Outcome labels, current architecture (submitted 2026-03-29)
- 500 games, n_sim=400, init from R4, no SF reeval — outcome labels (+1/-1/-0.1) directly
- Answers: do outcome labels improve geometry within the ReLU constraint?
- Expected: marginal improvement at best. ReLU cap is still in place.
- Scripts: `jobs/r5_selfplay.sh`, `r5_train.sh`, `r5_gate.sh`

### R6 — Tanh bottleneck, retrain from scratch (next after R5 results)
- One architectural change: `nn.ReLU()` → `nn.Tanh()` in bottleneck (done)
- Retrain from scratch — do NOT init from R4/R5. ReLU weights are geometrically wrong for Tanh.
- Outcome labels as primary signal
- Success criteria:
  - KQ vs K AND K vs KQ both classify correctly
  - Centroid cosine sim < 0.80
  - Separation gap > 0.057 (exceeds R2 peak)
  - Dead dimensions < 5/128

If R6 passes: the geometry hypothesis is alive. If R6 fails with Tanh: the CNN backbone itself may not be encoding material asymmetry in a geometry-compatible way — deeper architectural question.

### R7 — Geometric MCTS (only if R6 proves antipodal geometry)
Moves as trajectories in geometry space. Per-move MCTS bonus:

```
score(move) = value(board_after) + λ · Δg · (c_win − c_loss)
```

`Δg = geometry(board_after) − geometry(board_before)` projected onto the win-loss axis.
This gives MCTS a dense gradient at every node, not just leaf-node evaluation.
λ is tunable — start at 0.1.

### Hard decision after R7
After R7 we have enough signal to decide: does the geometry hypothesis hold at this scale? If yes, the path to GPU is justified (batched MCTS, larger model). If no, we reconsider the architecture more fundamentally (contrastive loss on the bottleneck, larger bottleneck, sequence model for position history). No premature commitment.

---

## Alternative Path — Geometry From First Principles (2026-03-30)

Identified during R6 analysis as a parallel direction, independent of whether R6/R7 succeed.

### The passenger problem

The geometry is currently shaped entirely by backprop through the value and policy heads. If the value head can produce correct scalar outputs without the geometry being meaningful, it will — and it does (R4: KQ vs K passes sanity check, fails geometry probe). The loss function contains no term that enforces antipodal structure. MCTS never touches the geometry directly. The geometry is a passenger: it goes along for the ride and occasionally picks up structure, but nothing forces it to be the driver of good play.

Making geometry a driver requires changing what it is optimised *for*, not just its activation function.

### The compute reality

We are not Google. Full-game self-play at scale requires GPU. The geometry hypothesis does not. Endgames are fully enumerable on CPU in minutes. The cheapest possible test of the geometry hypothesis is also the most principled one.

### Endgame-to-opening geometry curriculum

Endgames are the only part of chess where labels are unambiguous and positions are enumerable:

- **KQ vs K**: ~150k legal positions. All are either forced mate or stalemate. Label with Stockfish in minutes — perfect supervision. The geometry *must* be antipodal here or the model is simply wrong. No value head compensation is possible against a perfectly clean signal.
- **KR vs K**: similar scale, similar clarity.
- **K+P vs K**: slightly more complex, still tractable.
- **Pawnless midgame → full midgame → openings**: each step adds geometric complexity on top of a foundation that is already geometrically anchored.

This directly addresses the passenger problem. At each stage, the geometry has no noise to hide behind. Win/loss centroids are forced apart by the training signal, not hoped to emerge as a side effect.

The additional benefit: endgame geometry is the *ground truth* of chess value. A model that understands endgame geometry correctly has the right foundation for middlegame evaluation. Building openings on top of that is reasonable. Building endgame evaluation on top of opening intuition (the current approach) is backwards.

### Simplified chess for hypothesis testing

Before committing HPC resources to full-chess geometry experiments, test the geometry MCTS hypothesis on a toy domain where everything is verifiable:

- **4×4 or 5×5 board**, two kings and a small piece set (e.g., queen + pawns)
- Fully enumerable game tree — perfect labels, no Stockfish needed
- Self-play runs in minutes on a laptop
- Geometry is checkable against known perfect play

If geometry-based MCTS does not outperform standard MCTS on this toy problem, the hypothesis needs rethinking before scaling. If it does, you have proof of concept at negligible compute cost.

This is the Phase 0 lesson applied correctly: test the hypothesis in the cheapest possible domain, build infrastructure only after the idea is validated.

### Geometry transition function

For geometry MCTS to work beyond depth-1 (single legal move → geometry), a transition function is needed: given geometry g and move m, predict g' without a full board forward pass. This is a supervised learning problem once the geometry space is stable:

```
f(g_t, move) → g_{t+1}
```

Training data comes for free from every self-play game — you have the geometry at every position and every move played. The transition function is what enables planning in geometry space without expanding the full board tree. Legal move generation remains board-grounded; everything else is geometry.

### Per-piece geometry

Each piece carries its own geometry vector rather than contributing anonymously to a global board vector. The board geometry is a composition of piece geometries. This makes piece value ordering explicit and learnable from pure endgames — a queen's geometry activates more winning components than a rook's by construction, because they are trained on positions where that is unambiguously true.

Per-piece geometry also makes the geometry transition function tractable: when a piece moves, only that piece's geometry vector needs updating. The interaction term (board geometry = f(piece geometries)) can recompute cheaply from the changed piece rather than requiring a full board forward pass.

### Piece interference geometry (excluding the knight)

Each sliding piece (rook, bishop, queen) and the king and pawns have their reach constrained by other pieces on the board. A rook on an open file has different geometry than a rook blocked by its own pawn — not just different mobility, but a different positional identity. This constraint is part of the piece's geometry.

The knight is explicitly excluded. The knight jumps — its movement is not blocked by intervening pieces. Its geometry is determined by its square and the board boundaries alone, not by what stands between it and its destination. Treating the knight like a sliding piece here would be architecturally wrong.

Concretely: the interference geometry for a sliding piece is the set of squares it actually controls given current board occupancy (its ray truncated at blockers). This is a per-piece, per-position feature that feeds into the piece's geometry vector. It encodes both activity (open lines) and confinement (blocked lines) directly.

This is where back rank mate becomes representable: a rook's interference geometry pointing unobstructed at the opponent king's back rank, combined with the king's confinement geometry (pawns blocking escape squares), produces the danger signal. Neither alone is sufficient — the combination is.

### Relationship to R6/R7

R6 confirmed that changing the activation function alone (ReLU → Tanh) does not produce antipodal geometry — centroid cosine worsened from 0.869 to 0.981. The training signal never demanded geometric separation. R7 full-game selfplay is superseded by the endgame-first curriculum as the primary geometry path. R4 remains the best playing model.

---

## Session 7 — R6 Tanh Bootstrap (2026-03-30)

### Context

R5 selfplay completed (500 games, n_sim=400) but gate results were inconclusive or a regression — R4 remains the best confirmed model. R6 proceeds as planned: Tanh bottleneck, cold start, SF-labeled R5 data as bootstrap signal.

### R6 pipeline deviations from plan

The R6 plan called for outcome labels as primary signal. In practice, R6 uses SF-labeled R5 selfplay data (`selfplay_r6_sf.pt`) for the bootstrap step. This is a deliberate choice: a cold-start model needs accurate labels to orient the geometry — outcome labels on a random initialisation produce too much noise to establish the win/loss axis. SF labels at depth 20 give sharper signal for the bootstrap.

### Dataset concern

The R5 selfplay dataset had only **5,782 positions** (12 positions/game cap × 500 games ≈ 6k). ZIGZAG.md assumes ~30k per round. This is a known design mismatch — `MAX_POSITIONS_PER_GAME = 12` in selfplay.py was never revisited after the Lichess pretraining convention was carried over. R6 bootstrap and R6 selfplay are both operating at ~6k positions. For R7: increase `MAX_POSITIONS_PER_GAME` to 50 and reduce reeval depth to 15 (currently 2.8s/pos at depth 20 → ~18h for 23k positions; depth 15 brings it to ~4h).

### Reeval

SF depth 20 on 5,782 positions took **~4.5h** on a slow node. Three runs required — first two were wall-time kills (3h limit), third succeeded with 5h wall.

Label distribution: mean=+0.091, std=0.821, **85% decisive**. The high decisive rate reflects that R5 selfplay games played through to clear outcomes — late-game positions dominate. This is a harder distribution to train on than the Lichess bootstrap (std=0.565, 43% decisive).

### Bootstrap training

Trained from scratch on 5,493/289 train/val split. Early stopped at epoch 12, best at epoch 7. Val loss bottomed at 7.14 — training diverged after that, classic overfitting on a tiny dataset. Total training time: **126 seconds**.

Post-training sanity checks:

```
✓ White up queen          value=+0.932  (correct)
✗ Black up queen          value=+0.667  (wrong sign — expected negative)
✓ KQ vs K, White to move  value=+0.850  (correct)
✓ KQ vs K, Black to move  value=-0.478  (correct)
```

R4 also passed KQ vs K Black (-0.909) and Black up queen (-0.918) — all four checks clean. R6 is actually *worse* on the sanity checks than R4: it fails Black up queen where R4 didn't. The R4 geometry probe showed K vs KQ as ✗ FAIL at the representation level, but the value head compensated to give the correct sign. R6 hasn't yet learned to compensate.

The "Black up queen" failure root cause: the value head contains a `ReLU` between its two linear layers. ReLU breaks antisymmetry: `ReLU(-x) ≠ -ReLU(x)`. The KQ vs K check passes because the position is extreme enough to overcome this bias; the "Black up queen" check is more ambiguous and the ReLU asymmetry dominates. R4 avoided this because its weights were inherited from R2 — the accumulated learning gave the value head a better prior. R6 cold-start has not yet developed that compensation.

Fix identified for R7: replace `ReLU` with `Tanh` in the value head. One-line change in model.py. Deferred until R6 gate confirms the Tanh bottleneck is producing useful signal.

### Gate expectations

R6 is gated against R4. The comparison is asymmetric — R4 was fine-tuned from R2 over multiple rounds; R6 is a cold start on 5,782 positions. The gate threshold for advancing is not a clean win — any signal above 50% (even marginal) on 100 games confirms the Tanh architecture can orient itself from scratch. Clear improvement (>55%) would justify the full R7 treatment: larger dataset, depth 15 reeval, value head fix.

Gate results pending (scheduled for Tue Mar 31 ~04:00–08:00).

### R6 geometry probe

To run after gate: `compare_geometry.py` with R6 vs R2/R4 on `selfplay_r1_full_sf.pt`. Primary metrics to watch:
- Centroid cosine sim — target < 0.80 (vs R4: 0.869)
- Separation gap — target > 0.057 (vs R4: 0.048, R2 peak: 0.057)
- Dead dimensions — target < 5/128 (vs R4: 26/128)
- KQ vs K + K vs KQ both correct — partial pass already (K vs KQ now works)

---

## Next

1. R6 gate results (Tue Mar 31 morning)
2. R6 geometry probe — does Tanh bottleneck move the separation gap?
3. If signal present: R7 with value head fix + larger dataset
4. After R7: hard go/no-go on geometry hypothesis

---

## Session 8 — ⟳ DOOVER 2: Endgame Curriculum + SF Lichess (2026-04-09)

*Rounds reset. Previous best model (R4, 67% vs material) retired.*
*Motivation: ReLU bottleneck prevents antipodal geometry. New approach: Tanh bottleneck + endgame curriculum + SF-Lichess broad training.*

### Round 1 — Endgame curriculum (stage 1+2)

Trained from scratch (Tanh bottleneck) on 20k KQ vs K + KR vs K positions with antipodal mirrors, 20 epochs, value-only (policy weight=0), regenerating positions each epoch to prevent memorisation.

**Geometry probe results:**

| Metric | Result | vs R4 |
|--------|--------|-------|
| Centroid cosine | **-0.9999** | 0.869 → essentially perfect |
| Separation gap | **1.9971** | 0.048 → theoretical max is 2.0 |
| Effective rank | **1.0 / 128** | COLLAPSED — 1D |
| KQ vs K (W2M) | ✓ +0.983 | ✓ |
| KQ vs K (B2M) | ✓ -0.984 | ✓ |
| Black queen up | ✓ -0.984 | ✗ (R6 failed) |
| NN consistency | 1.000 | 0.902 |

**Interpretation:** The geometry mechanism works — antipodal separation is achievable and the endgame curriculum forces it cleanly. However, the training signal is binary (one side always has the winning piece, no draws, no gradations) so the model learned a single win/loss axis. Effective rank 1.0/128 means 127 dimensions are unused. The centroid cosine and separation gap look perfect but are trivially so — two perfectly separated clusters with no internal variation.

**Decision:** Skip the remaining endgame stages (3–8). They would expand rank from 1 to maybe 5 but the distribution is still too narrow. The correct next step is SF-labeled Lichess data with continuous labels — hundreds of material configurations simultaneously — mixed with endgame anchors (15%) to keep the win/loss axis grounded.

### Round 2 — sf_balanced (SF-Lichess + endgame anchors)

*Model: `models/sf_balanced/best.pt`*

Dataset: `dataset_sf.pt` (max_pieces=20, decisive) merged with `dataset_balanced.pt` (max_pieces=32, min_decisive=0.05) + 15% endgame anchors (KQK + KRK).

**Results:**

| Metric | Value |
|--------|-------|
| Step 2 Greedy vs Random | **75%** ✓ |
| Step 5 vs MCTS(material) | **58% (+56 ELO)** ✓ |
| Effective rank | **7.2 / 128** ✗ (target >30) |
| Centroid cosine (win·loss) | **0.0802** ✓ (was 0.869 in R4) |
| Separation gap | **0.2014** ✓ (was 0.048 in R4) |
| Draw cluster (win·draw) | 0.2657 — draws loss-adjacent, no separate axis |
| β1 topology loops | 121 — healthy non-trivial structure |
| Geometry MCTS (Step 6) | **29.2%** ✗ — passenger problem persists |

All sanity checks pass. Starting position mislabelled as "loss" — caused by max_pieces=20 training making full-board positions OOD.

### Doover architecture decisions confirmed
- **Tanh bottleneck**: confirmed correct, antipodal geometry achievable
- **ReLU value head**: leave as-is — passes antisymmetry with balanced training data
- **GPU**: switching to `gpuv100` queue for training (24h wall limit)

### Dataset pipeline (in progress 2026-04-09)
- Lichess Jan 2025 PGN already on HPC (11GB)
- `parse_lichess` job submitted — 150k games, min ELO 1500, `--no-strict` (Check 6 false positive due to STM-relative representation)
- SF reeval job ready to submit after parse completes
- Training job: SF dataset primary + 15% endgame stage 1+2 anchors

### Monitoring current parse job

```bash
bjobs                                          # check status
tail -f ~/logs/parse_lichess_<JOBID>.out       # watch progress (prints every 10k games)
cat ~/logs/parse_lichess_<JOBID>.err           # check errors
ls -lh ~/Petra-ver2/data/dataset.pt            # confirm output exists when done
```

### Next steps (in order)

```bash
# 1. After parse completes — submit SF reeval
bsub < ~/Petra-ver2/jobs/reeval_sf.sh

# 2. After reeval completes — submit GPU training
bsub < ~/Petra-ver2/jobs/train_sf_gpu.sh      # (to be written)

# 3. After training — run geometry probe
python3 src/generate_endgame.py --positions 5000 --stages 1 2 --out ~/Petra-ver2/data/endgame_probe.pt
python3 src/probe_geometry.py --model ~/Petra-ver2/models/sf_gpu/best.pt --dataset ~/Petra-ver2/data/endgame_probe.pt --n 5000
```

**Gate criterion:** effective rank > 30. Centroid cosine and separation gap are secondary — they were trivially perfect on 1D geometry and will naturally drop as rank expands. That is expected and correct.

---

## Session 9 — Geometry analysis + Round 3/4 plan (2026-04-15)

### Round 3 — zigzag r1 (self-play from sf_balanced)

*Model: `models/zigzag/r1/best.pt`*

One round of self-play (500 games, n_sim=40, SF depth 12) starting from sf_balanced. Opening book used to skip biased first moves. 20% endgame anchor mix during training.

**Results:**

| Metric | Value |
|--------|-------|
| Step 5 vs sf_balanced (head-to-head) | **51.2% (+9 ELO)** |
| Step 5 vs MCTS(material) | **~59% est.** |
| Geometry MCTS (Step 6) | **28.0%** ✗ — passenger problem unchanged |
| Probe R² (geometry→outcome) | 0.84 — geometry has signal but not directional |

Self-play did not regress. Gain is marginal (+9 ELO over sf_balanced). Geometry rank not probed separately; Step 6 result confirms no improvement in directional structure.

**Diagnosis:** Rank 7.2 is the bottleneck. The value head compensates for weak geometry, so self-play has no pressure to improve it. More rounds of zigzag from this base are not worth running until geometry rank improves.

### Round 4 — train_round2 (full pieces + rank regularisation) — PENDING

*Model: `models/round2/best.pt` (not yet trained)*

*Script: `jobs/train_round2_gpu.sh`*

Changes from Round 2 (sf_balanced):
- Primary dataset: `dataset_balanced.pt` only (max_pieces=32, draws included) — no max_pieces=20 restriction. Fixes starting-position OOD problem.
- Cold start — sf_balanced's 7D subspace may be a local attractor.
- `--rank-reg 0.1`: adds λ·tr(C²) to loss. At rank=7.2 this adds ~0.014/batch — direct gradient toward rank expansion. Also pushes draw cluster off win/loss axis.
- `--anchor-frac 0.20` (up from 0.15) to keep win/loss axis oriented as rank expands.

**Gates:**
- Step 2 Greedy vs Random > 70% (full-board training must not break policy)
- Step 5 vs MCTS(material) > 55% (baseline: sf_balanced at 58%)
- Effective rank > 15 (stretch: > 30)
- win·draw cosine should drop below 0.20 (draw dimension opening)

*Run order: `bsub < jobs/train_round2_gpu.sh` → probe_geometry → eval_round2_vs_material*

### Code changes this session

- `train.py`: `--rank-reg λ` flag. Adds λ·tr(C²) loss term via `_geometry_fwd()`. RankL column in training output.
- `evaluate.py`: self-describing header printed at start of every eval log (model, steps, games, n_sim, opponent, date).
- `reeval_stockfish.py`: clarifying comments that UCI `score cp` is already STM-relative — no flip needed or correct. Distinct from STM fixes in board.py/data.py.
- New jobs: `eval_round2_vs_material.sh`, `train_round2_gpu.sh`.

---

## Milestones

### ELO
- [x] Beat MCTS(material) at >55% — *R2: 61%*
- [x] Beat MCTS(material) at >60% — *R2: 61%, R4: 67%*
- [ ] Beat MCTS(material) at >70% — *R4: 67%, close*
- [ ] Beat Stockfish depth 1
- [ ] Beat Stockfish depth 5

### Geometry (the thesis)
- [x] Win/loss centroid cosine < 0.85 — *Doover 2 R2 (sf_balanced): 0.0802*
- [x] Win/loss centroid cosine < 0.00 — *Doover 2 R1 (endgame): -0.9999*
- [x] KQ vs K AND K vs KQ both correct — *Doover 2 R2 (sf_balanced): both ✓*
- [ ] Effective rank > 30 — *Doover 2 R2: 7.2/128*
- [ ] Draw dimension: win·draw cosine < 0.10 — *currently 0.2657*
- [ ] Separation gap > 0.10 with rank > 30 — *gap 0.2014 but rank still low*

### Self-play
- [x] 1k self-play positions trained
- [x] 10k self-play positions — *R1 full run*
- [x] 100k self-play positions — *R2–R4*
- [x] 1M MCTS simulations — *R4: n_sim=400 × 200 games*
- [ ] 10M MCTS simulations — *GPU territory*

---

## Session 10 — Fork: Decisiveness Zigzag (2026-04-20)

### Context

lichess_2023_03 trained on SF depth-18 labels showed geometry regression vs feb_sf
(rank 17.1 vs 18.9, loss·draw cosine 0.4550). Root cause: 2500+ ELO filter produces
a value distribution too compressed near zero — geometry has no signal to separate
win/draw/loss. Fix in progress: re-parse with 2000+ ELO, retrain from feb_sf.

The endgame-first curriculum (Doover 2 Round 1) was dropped — it collapsed rank to
1.0. The position-type chunking idea evolved into a simpler approach that uses only
real game positions already in the SF-labeled dataset.

### Proposed: Decisiveness Zigzag

Instead of synthetic endgames or separate position-type datasets, zigzag between
decisiveness-filtered subsets of the same SF-labeled dataset:

```
Stage 1: train on |v| > 0.7  (most decisive)   → forces win/loss axis apart
Stage 2: train on |v| > 0.5  (decisive)         → expands axis, adds gradation
Stage 3: train on all positions                  → full rank expansion with draws
```

Each stage fine-tunes from the previous. Decisive positions come from real sampled
games — no artificial generation needed. The SF-reeval'd dataset already has
continuous labels so filtering is a threshold, not a new pipeline step.

**Why this avoids past failures:**
- Pure endgame training (Doover 2 R1): collapsed to 1D because binary labels
- Pure 2500-ELO SF training (lichess_2023_03): compressed because too few decisive
- Decisiveness zigzag: starts sharp (geometry forced apart), then broadens (rank expands)

**Gate at each stage:** effective rank must increase before advancing to the next.
If rank stalls between stages → step size is too large, add an intermediate threshold.

### Parse strategy revision (2026-04-20)

Hard-filtering by piece count (decisive endgames only) was rejected — tried previously
and collapses complexity. The correct approach is a **slow staircase via ELO**:
parse at MIN_ELO=1850, which naturally mixes decisive and drawn positions across all
game phases. Lower ELO → more blunders/material imbalances → SF labels spread across
full range without removing the breadth geometry also needs.

Job: `parse_month_2200.sh` (name is a historical artefact — it now defaults to 1850).

### SF label distribution — old vs new dataset

Measured on `dataset_2023_03_sf18.pt` (2500 ELO, OOM-killed at 80k/100k games):

| Metric | Value |
|--------|-------|
| Total positions | 284,866 |
| Near-zero \|v\|<0.1 | 87,239 (30.6%) |
| Decisive \|v\|>0.7 | 71,982 (25.3%) |

30.6% of positions were essentially balanced — geometry had no signal to push
win/loss apart. Combined with the tiny dataset (OOM cut it to 285k), this explains
both the rank regression (17.1 vs 18.9) and ELO drop.

Target after 1850 ELO reeval: near-zero <20%, decisive >35%. Run same check after
merge to confirm before training.

### Fork condition (eval pending 2026-04-20)

lichess_2023_03 vs feb_sf head-to-head result determines priority:

- **lichess_2023_03 wins >50%**: geometry regression is noise for ELO; proceed with
  re-parse (2000+ ELO) + retrain from feb_sf, then add decisiveness zigzag on top.
- **lichess_2023_03 loses**: geometry regression actively hurts ELO; decisiveness
  zigzag is critical path, not optional.

---

## Theoretical Section — Geometry Architecture (2026-04-20)

### What L2 normalization can and cannot do

L2 normalization constrains all embeddings to a unit hypersphere. In the best case
this produces a clean three-way separation of win/draw/loss centroids. Realistically,
the most likely outcome is a win/loss hyperplane — draw is harder to anchor because
drawn positions are rarer and less decisive in the training signal.

This is still extremely valuable as a proof of concept: a learned win/loss axis on a
unit sphere, driven by SF labels, is a non-trivial geometric structure. The ceiling
of L2 is dimensional separation — a hypersphere is the wrong space for the full
richness of chess positional structure, but it is the right place to start.

**Current strategy:** push L2 separation as far as possible (rank, cosine gates,
staircase training), and probe topology as thoroughly as possible. Establish what
L2 can achieve before replacing it.

### Topology as the deeper signal

Rank and centroid cosines measure whether the manifold is spread out. Topology
measures whether it has the right *shape*. A rank-30 manifold can still be
topologically trivial — one undifferentiated blob. Topology probing (β0, β1,
persistence) should be treated as a primary gate alongside rank, not a secondary
diagnostic.

Open questions to resolve with current architecture before moving on:
- Do win/draw/loss form topologically distinct regions or a single gradient?
- Are game transitions locally consistent (adjacent positions close in the manifold)?
- Does β1 reflect chess concepts (repetition, fortress, zugzwang) or noise?
- How does topology evolve across staircase stages?

### Future direction — Hyperbolic geometry (Poincaré embeddings)

The intuition: chess positional structure is hierarchical. Material imbalance dominates,
then pawn structure, then king safety, then tempo. Hyperbolic space represents
hierarchies naturally — exponentially more room near the boundary, richer capacity
for nested concepts than a flat sphere.

A strict staircase is even more critical in hyperbolic space: each level of the
hierarchy must be introduced without collapsing or dominating the levels already
learned. The risk is that a dominant concept (e.g. material) fills the entire
hyperbolic disk, leaving no room for subtler structure to develop.

**This is a future phase.** Current work exhausts L2 first. The transition to
Poincaré embeddings is only warranted once L2 geometry has been pushed to its
ceiling and the ceiling is measured.

### Topological concept probes — ideas (implement later)

These probes are only meaningful once two preconditions are met:

- **ELO ≥ 1700**: below this, geometry likely reflects blunder detection rather
  than positional concepts. A probe that fires on "side with more material wins"
  is not a concept probe.
- **Effective rank ≥ 30**: fewer dimensions means concept probes risk fitting to
  collapsed or random dimensions rather than structured representations. Results
  would not be interpretable.

Probe ideas, roughly ordered by implementation cost:

1. **Linear concept probes** — label positions for a specific concept (passed pawn,
   bishop pair, open file, king exposure) and train a linear classifier on bottleneck
   embeddings. Success = concept is geometrically encoded and linearly separable.
2. **Transition consistency** — embed full game sequences and measure path smoothness.
   Adjacent positions via legal moves should trace continuous trajectories. Jumpy paths
   mean geometry-driven search would be incoherent.
3. **Loop identification (β1 → chess meaning)** — sample positions sitting on
   persistent cycles. Do they correspond to repetition draws, fortress structures,
   zugzwang? Connects measured topology to actual chess logic.
4. **Phase clustering** — partition by game phase (opening/middlegame/endgame) and
   measure topological distance between clouds. Tests whether the manifold has learned
   game structure or just value.
5. **UMAP/t-SNE coloured by concept** — qualitative, project 128→2D and colour by
   material balance, phase, or outcome. Visual sanity check before committing to
   quantitative probes.

### Draw representation — beyond 2D (2026-04-21)

**The core problem:** Draw is not a positional property — it is a dynamical one.
Win and loss describe the state of the board. Draw describes the *trajectory* of
the game: under optimal play, the position does not progress toward resolution.
It cycles. A fortress loops indefinitely. Repetition is a literal periodic orbit.
KR vs KR has no winning direction — it oscillates.

This means win/draw/loss cannot be cleanly represented as three linear directions
in a shared geometry space. Win and loss are antipodal on an axis. Draw is not a
third pole — it is a different kind of object. Placing a draw centroid perpendicular
to the win/loss axis (the draw perp loss approach) is a linear approximation of
something that is fundamentally non-linear.

**WDL head analysis:**

A WDL head with SF-derived labels does not solve this. The conversion
`p_draw = 1 - |v|` defines draw as *the complement of decisiveness* — not as
a genuine third category. A near-equal sharp middlegame and KR vs KR both get
`p_draw ≈ 1.0` but are geometrically unrelated. The three-way gradient pressure
from WDL is only meaningful if the draw label comes from actual game outcomes,
not scalar SF evaluations. With game-outcome WDL labels the draw signal is noisy
(blunders, ELO dependence) but structurally correct. With SF-derived WDL the
signal is clean but the draw concept is fake.

**What draw actually looks like mathematically:**

A draw trajectory in geometry space has small net displacement and oscillatory
character — the geometry vectors form a closed loop or remain in a bounded region.
Win trajectories move consistently toward one pole; loss trajectories toward the
other. The mathematical signature of draw is not a direction but a **winding number**
— a topological invariant counting how many times a trajectory loops. Win and loss
trajectories have winding number 0 (they go somewhere and stop). Draw trajectories
have non-zero winding number (they orbit).

The covariance structure of draw positions may also differ: not one dominant
eigenvector (a direction) but two or more comparable eigenvalues forming a plane —
a torus-like submanifold rather than a directed arc. The β1 loops detected in
persistent homology may literally correspond to draw cycles — probe idea #3
(sample positions on persistent β1 cycles: do they correspond to repetition,
fortress, zugzwang?) is the empirical test.

**Consequences for current architecture:**

The current architecture processes individual positions. Draw-ness as a dynamical
property requires either trajectory input (sequences of positions) or the model to
implicitly learn that certain position types precede draw trajectories. The endgame
pool (KR vs KR etc.) is the closest approximation — those positions are structurally
associated with draw dynamics even as point data. But it is an approximation of
the underlying topological truth.

**The clean mathematical decomposition (2026-04-21):**

Separate value from drawness explicitly:

```
q = P(win) - P(loss)     # value — side-to-move advantage, scalar in [-1, 1]
d = P(draw)              # drawness — confidence that the position is structurally drawn

P(win)  = (1 - d + q) / 2
P(loss) = (1 - d - q) / 2
P(draw) = d
```

This makes the distinction geometrically real:

| Position | P(win) | P(draw) | P(loss) | q | d |
|----------|--------|---------|---------|---|---|
| Dead draw (KR vs KR) | .02 | .96 | .02 | 0 | .96 |
| Sharp balanced | .45 | .10 | .45 | 0 | .10 |
| Winning | .85 | .10 | .05 | +.80 | .10 |

Both dead draw and sharp balanced have q=0 (value-equal) but d is completely different.
The geometry gets a genuine second signal that is orthogonal to value.

**Label semantics — balanced ≠ draw:**

The codebase and training pipelines have conflated these throughout. They must be
separated:

- `|v| < threshold` from SF labels → **balanced** (near-equal, could go either way)
- Structural/theoretical draw sources → **draw** (neither side can make progress)

Structural draw sources (ground truth for d):
- KR vs KR, KNN vs K, KB vs KB — endgame pool
- Insufficient material positions
- Tablebase-confirmed draws
- PGN games ending in draw *from positions already near-equal* (lower confidence)
- SF WDL draw probability if available at reeval time

The `|v| < 0.1` Lichess bucket is balanced, not drawn. Using it as a draw anchor
trains the model to treat "SF has no strong opinion" as draw-ness. That is wrong and
is the root cause of the draw dimension never opening cleanly.

**Implementation update (2026-04-22): explicit drawness supervision**

The near-term drawness head is now implemented as an auxiliary
`Linear(128→1) + sigmoid` head over the geometry bottleneck. Training no longer
has to infer drawness from `|v| ≈ 0`. Endgame datasets can carry:

- `drawness_targets`: `1.0` for structural draws, `0.0` for known non-draws
- `drawness_mask`: rows where that target is valid
- `drawness_available`: marks datasets that intentionally supplied drawness labels

`generate_endgame.py` now emits explicit drawness targets:

- stages `9..11` (`KR vs KR`, `KNN vs K`, `KB vs KB`) → drawness positive
- stages `1..8` decisive endgames → drawness negative

`train.py` prefers these explicit targets. Older anchors still fall back to the
previous heuristic (`is_anchor & |v| < 0.15` as positives, `|v| > 0.5` as
negatives), but new runs should use explicit masks. This prevents the critical
failure mode where balanced Lichess positions are mislabeled as theoretical draws.

**Training schedule: bootstrap → broaden → refresh**

Drawness should be bootstrapped before broad training, then periodically refreshed:

1. **Drawness bootstrap:** train from a competent checkpoint on generated/tablebase
   endgames with explicit drawness labels. Use structural draw stages as positives
   and decisive endgames as negatives. Policy can be disabled or downweighted.

   Example:

   ```bash
   python3 src/train.py \
     --endgame-positions 250000 \
     --endgame-stages 1 2 4 5 9 10 11 \
     --init-model models/feb_sf/best.pt \
     --draw-reg 0.05 \
     --rank-reg 0.05 \
     --policy-weight 0.0 \
     --epochs 5 \
     --out models/drawness_bootstrap
   ```

2. **Broader value/policy training:** switch to Lichess/SF/self-play data, using
   the bootstrapped model as `--init-model`. Keep a small structural draw anchor
   or low `--draw-reg` active so the drawness head does not drift while the value
   and policy geometry broaden.

3. **Drawness refresh:** after each broad phase, run a short endgame refresh
   with higher drawness weight. This re-anchors "neither side can make progress"
   without treating every equal middlegame as drawn.

Recommended first-pass cadence:

```
2-5 epochs drawness bootstrap
5-10 epochs broad SF/Lichess training with low drawness anchor
1-2 epochs drawness refresh
repeat
```

The drawness head replaces the old draw-perpendicular loss as the primary
anti-collapse mechanism. `L_draw = λ cos²(c_draw, axis)` remains a historical
experiment: useful as an anti-collapse probe, but it creates a fake single draw
axis if used as the main supervision.

**Decision path:**

1. **Near term:** Use the implemented auxiliary drawness head with explicit
   targets and the bootstrap/broaden/refresh curriculum above.

2. **Medium term:** WDL head with game-outcome labels combined with structural draw
   anchors. The q/d decomposition above is the target output. Noisy from blunders
   but structurally correct over large n.

3. **Long term:** once trajectory data is available (self-play games with per-move
   geometry embeddings), use path topology to detect draw — small geometry
   displacement, oscillatory character, low Lyapunov exponent of the geometry
   trajectory. Draw detection becomes topological rather than linear. The β1 probe
   (loop identification → chess meaning) is the empirical bridge to this stage.

**The decisive gate for draw understanding:**

Not whether draw vectors sit between win and loss. Whether Petra can represent
*"neither side can make progress"* separately from *"both sides have chances."*
Sanity test: KR vs KR and a sharp balanced middlegame (SF eval ≈ 0, many captures
available) should produce different drawness scores. Until that test passes,
draw geometry is not solved.

---

## Long-Term Phases

| Phase | Goal |
|-------|------|
| 0 | Geometry encoder on Stockfish values — **FAILED** |
| 1 | Supervised pretraining on Lichess outcomes (now) |
| 2 | Online geometry update from game outcomes |
| 3 | Self-play loop — geometry evolves from game experience |
| 4 | Full autonomy — Petra selects own curriculum |

**Success criterion:** Petra reaches ~2000 ELO vs Stockfish without hand-crafted heuristics.
