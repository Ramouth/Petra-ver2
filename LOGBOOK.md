# Petra Training Logbook

---

## Doover — Supervised Pretraining (2026-04-13)
*Job 28197059, GPU V100 (n-62-20-2), ~79s wall time*

### Dataset
| | |
|---|---|
| Source | Lichess positions re-evaluated with Stockfish depth 15 |
| Raw positions evaluated | 200,000 |
| Filter | `min_decisive=0.3`, `max_pieces=20` |
| Kept | 44,284 / 200,000 (22.1%) |
| Dropped (eval) | 81,916 |
| Dropped (pieces) | 73,800 |
| Post-filter decisive rate | 92.3% |
| Label mean / std | +0.056 / 0.576 |
| Train / val split | 42,137 / 2,147 |

### Anchor
| | |
|---|---|
| Dataset | `endgame_anchor.pt` — KQK + KRK, stages 1+2 |
| Positions | ~35,898 (10k base × mirrors) |
| Sampled | 6,320 (15% of train) |
| Mixed train size | 48,457 |
| Labels | Rule-based ±1.0, perfectly balanced |

### Training
| Epoch | T-loss | V-loss | V-MSE | V-R² | Top1 | Top5 | LR |
|-------|--------|--------|-------|------|------|------|----|
| 1 | 8.466 | 7.901 | 0.266 | 0.621 | 0.008 | 0.038 | 1e-3 |
| 2 | 7.650 | 7.483 | 0.237 | 0.662 | 0.008 | 0.047 | 1e-3 |
| 3 | 7.363 | 7.741 | 0.620 | 0.110 | 0.008 | 0.042 | 1e-3 |
| 4 | 7.239 | 7.474 | 0.442 | 0.360 | 0.009 | 0.043 | 1e-3 |
| 5 | 7.164 | 7.166 | 0.206 | 0.707 | 0.011 | 0.043 | 1e-3 |
| 6 | 7.116 | 7.332 | 0.365 | 0.473 | 0.011 | 0.045 | 1e-3 |
| 7 | 7.079 | 7.729 | 0.782 | -0.134 | 0.010 | 0.041 | 1e-3 |
| 8 | 7.030 | 7.421 | 0.533 | 0.228 | 0.011 | 0.051 | 5e-4 |
| 9 | 6.963 | 7.027 | 0.187 | 0.735 | 0.011 | 0.053 | 5e-4 |
| 10 | 6.926 | 7.033 | 0.224 | 0.670 | 0.014 | 0.058 | 5e-4 |
| 11 | 6.884 | 7.132 | 0.342 | 0.503 | 0.012 | 0.055 | 5e-4 |
| **12** ★ | **6.844** | **6.947** | 0.189 | 0.726 | 0.013 | 0.069 | 5e-4 |
| 13 | 6.796 | 7.115 | 0.353 | 0.492 | 0.016 | 0.075 | 5e-4 |
| 14 | 6.748 | 7.047 | 0.328 | 0.526 | 0.018 | 0.080 | 5e-4 |
| 15 | 6.701 | 7.095 | 0.413 | 0.408 | 0.022 | 0.088 | 2.5e-4 |

★ Best checkpoint. Early stopping patience not triggered — all 15 epochs completed.

### Sanity checks
| Position | Value | Pass |
|---|---|---|
| Start position | -0.168 | ~ |
| White up queen | +0.912 | ✓ |
| Black up queen | -0.905 | ✓ |
| KQ vs K, White to move | +0.831 | ✓ |
| KQ vs K, Black to move | -0.823 | ✓ |

**Note on start position:** -0.168 is a meaningful black lean, not noise. Root cause: `max_pieces=20` filter excluded all opening positions from training — the full board (32 pieces) is out-of-distribution. Will self-correct via zigzag self-play games. Next reeval should use `max_pieces=28–30` to include opening positions.

### Model
`models/sf_gpu/best.pt` — 1,357,057 parameters

---

## Geometry Probe — Baseline (2026-04-13, Job 28199994)
*Before zigzag round 1. Model: `sf_gpu/best.pt`. Dataset: `dataset_sf.pt`, n=5000.*

| Metric | Value | R4 historical | Notes |
|--------|-------|---------------|-------|
| Effective rank | **3.7 / 128** | ~102 active dims (26 dead) | Far below healthy target of >30 — almost pure win/loss axis |
| Win·Loss centroid cosine | **0.6665** | 0.869 | Much better separation than R4 |
| Separation gap | **0.2427** | 0.048 | 5× better than R4 — geometry is genuinely structured |
| NN label consistency | **0.808** (lift +0.375) | — | Nearby positions share labels reliably |
| Win·Draw cosine | 0.8413 | — | |
| Loss·Draw cosine | 0.9368 | — | Draws cluster near losses — model has weak draw concept |

**Raw output:**
```
============================================================
CHECK 1 — Eigenvalue distribution
============================================================
  Top-1  eigenvalue: 43.8% of variance
  Top-5  eigenvalues: 89.8% of variance
  Top-10 eigenvalues: 98.0% of variance
  Effective rank: 3.7 / 128  (healthy > 30, collapsed < 10)
  PARTIAL — some concentration but not collapsed

CHECK 2 — Win / draw / loss separation
  Samples — win: 1078, draw: 173, loss: 896
  Centroid cosine similarities:
    win  · loss  = 0.6665
    win  · draw  = 0.8413
    loss · draw  = 0.9368
  Within-class cosine similarity:
    win  pairs: 0.6244
    loss pairs: 0.7477
  Between-class (win vs loss): 0.4434
  Separation gap: 0.2427
  GOOD — win/loss positions occupy distinct regions

CHECK 3 — Known position probes
  Starting position              →win 0.666  →loss 0.814  value -0.168  → loss
  KQ vs K — White to move        →win 0.760  →loss 0.469  value +0.905  → win
  KQ vs K — Black to move        →win 0.491  →loss 0.947  value -0.899  → loss
  Equal endgame (KR vs KR)       →win 0.780  →loss 0.450  value +0.928  → win  ← WRONG
  White queen up                 →win 0.711  →loss 0.408  value +0.912  → win
  Black queen up                 →win 0.051  →loss 0.305  value -0.905  → loss
  Complex middlegame             →win 0.623  →loss 0.887  value -0.580  → loss

CHECK 4 — Nearest-neighbour label consistency
  Mean NN label match rate : 0.808
  Random baseline          : 0.433
  Lift                     : +0.375
  GOOD — nearest neighbours share labels more than chance
```

**Interpretation:** Strong win/loss separation but geometry is effectively 1-dimensional. The model knows who's winning but has no middle ground. Root cause: 92.3% decisive training data — model has almost never seen a balanced position. KR vs KR equal endgame scored +0.928 (should be ~0) confirms no draw understanding.

---

## Evaluation vs Material (2026-04-13, Job 28200021)
*Model: `sf_gpu/best.pt`. 100 games per step, n_sim=200, 16 workers.*

| Step | Matchup | W | D | L | Win rate | ELO Δ | Result |
|------|---------|---|---|---|----------|-------|--------|
| 2 | Greedy(policy) vs Random | 0 | 50 | 50 | 25.0% | -191 | FAIL |
| 5 | MCTS(learned) vs MCTS(material) | 58 | 18 | 24 | 67.0% | +123 | **PASS** |
| 6 | MCTS(geometry) vs MCTS(material) | — | — | — | — | — | SKIP (--probe-dataset missing) |

**Raw output:**
```
--- Step 2: Policy check (Greedy vs Random) ---
  Games : 100  (W=0 D=50 L=50)
  Score : 25.0/100  (25.0%)
  ELO Δ : -191  (95% CI: [17.5%, 34.3%])
  Step 2 [FAIL]  Policy check  wr=25.0%  ELO Δ=-191

--- Step 5: Learned value (gate) (MCTS(learned) vs MCTS(material)) ---
  Games : 100  (W=58 D=18 L=24)
  Score : 67.0/100  (67.0%)
  ELO Δ : +123  (95% CI: [57.3%, 75.4%])
  Step 5 [PASS]  Learned value (gate)  wr=67.0%  ELO Δ=+123
  GATE PASSED — learned value beats material. Proceed to self-play.

--- Step 6: Geometry value ---
  SKIP: --probe-dataset required for step 6
```

**Step 5 gate passed** — 67% exceeds the 55% threshold. Proceed to self-play.

**Step 2 failure — policy concern (open question):**
The perfectly regular W=0 D=50 L=50 result (every 10-game checkpoint: exactly W=0, D=5n, L=5n) is anomalous. One hypothesis: the policy is degenerate and always picks the same move, causing repetition draws as white and systematic losses as black. Supporting evidence: policy top-1 accuracy was only 0.022 in training.

However, the 67% step 5 result challenges a strong degeneracy claim — if the policy were truly broken, MCTS would waste simulations on one branch and performance would suffer. More likely the policy is just very weak but not pathologically degenerate, and the value head (separation gap 0.2427) is primarily carrying the +123 ELO advantage.

**Concern for zigzag:** A weak policy prior in MCTS could reduce self-play diversity — search concentrating on narrow lines → correlated training data → potential passenger problem. Severity unclear. Decision deferred until `reeval_balanced` completes and the next training run can be evaluated. Model saved at `sf_gpu/best.pt` so any round 1 outcome is recoverable.

*Step 6 fix needed: add `--probe-dataset` to `eval_sf.sh` before next run.*

---

## Comparison to Previous Rounds

| Run | Dataset | Train size | Best V-R² | Black up queen | Start value | Notes |
|-----|---------|-----------|-----------|----------------|-------------|-------|
| R4 | Self-play (R4) | ? | ? | ✓ | ? | Best confirmed model pre-doover. Geometry: cosine 0.869, gap 0.048, 26/128 dead |
| R6 | Self-play (R5) | 5,782 | ? | ✗ (+0.667) | ? | Cold start, tiny dataset, value head ReLU bug. Best epoch 7, early stopped ep 12 |
| **Doover** | SF-labeled Lichess + endgame anchor | 48,457 | **0.735** | **✓ (-0.905)** | -0.168 | All sign checks pass. OOD opening bias noted |

**Key improvements vs R6:**
- Black up queen now correct (value head ReLU→Tanh fix confirmed working)
- 8× more training data (48k vs 5.8k)
- Depth 15 SF labels vs self-play outcome labels
- Endgame anchor preventing passenger problem

**Geometry probe pending** — baseline measurement against R4 (cosine 0.869, gap 0.048) scheduled before zigzag round 1.

---

---

## Bug: Missing STM Flip in reeval_stockfish.py (discovered 2026-04-14)

### Root Cause
`reeval_stockfish.py` `evaluate()` returns `tanh(cp / 400)` directly from Stockfish output. Stockfish CP scores are **White-relative** (positive = White better). The method docstring claims "from the perspective of the side to move" but the STM flip was never implemented.

When Black is to move, the label sign should be flipped. Without the flip, Black-to-move positions are labelled as if White is winning — introducing a systematic white-advantage bias.

### Why It Wasn't Caught Earlier
`dataset_sf.pt` (used to train `sf_gpu/best.pt`) used `min_decisive=0.3, max_pieces=20`. Heavily decisive late-game positions have a dominant win/loss signal that swamps the sign error — the bias appeared as a mild -0.168 start position value.

`dataset_balanced.pt` used `min_decisive=0.05, max_pieces=32`, keeping full-board and balanced positions where the STM flip matters most. The error surfaced clearly: `sf_balanced/best.pt` start position = **+0.299**.

### Affected Datasets
Both `dataset_sf.pt` and `dataset_balanced.pt` were generated with the buggy script. Both must be regenerated after the fix.

### Fix
In `src/reeval_stockfish.py`, `evaluate()`:
```python
if mate is not None:
    val = 1.0 if mate > 0 else -1.0
    return val if fen.split()[1] == 'w' else -val
if cp is not None:
    val = math.tanh(cp / 400.0)
    return val if fen.split()[1] == 'w' else -val
return 0.0
```

### Next Steps

### Currently running
- **Job 28200694** `train_balanced_gpu` — completed (tainted; `sf_balanced/best.pt` trained on mislabelled data, start value +0.299)
- **eval_sf** — gates running against `sf_balanced/best.pt` (results pending)

### After gates finish
1. Apply STM fix to `src/reeval_stockfish.py`
2. Re-run `reeval_sf` to regenerate `dataset_sf.pt` with correct labels
3. Re-run `reeval_balanced` to regenerate `dataset_balanced.pt` with correct labels
4. Re-run `train_balanced_gpu` — merged training with fixed datasets
5. Geometry probe on new model
6. Update `zigzag_r1.sh` seed model and submit

### Why the extra training run
`sf_gpu/best.pt` has effective rank 3.7/128 — almost pure win/loss axis, no draw concept, opening positions OOD (trained on max_pieces=20 only). `dataset_balanced.pt` adds full-board positions and balanced labels to fill the geometry gap before self-play.
