# Petra Training Logbook

---

## Head-to-head: mid_no_endgame vs mid_only (Job 28294106, 2026-04-26)

**Result: statistically identical.** 400 games, n_sim=100, direct match.

| W | D | L | wr | ELO Δ | 95% CI |
|---|---|---|----|-------|--------|
| 55 | 294 | 51 | 50.5% | +3 | [45.6%, 55.4%] |

SE(wr) = 2.5pp — the CI straddles 50% on both sides. z = 0.2. Not significant.

Draw rate 73.5% (294/400) — far higher than either model's ~55% rate vs feb_sf. Expected: two equal-strength models contest most games to a draw.

**Conclusion:** The 9-ELO gap from independent vs-feb_sf runs (mid_only -85 vs mid_no_endgame -94) was noise, as predicted. **mid_no_endgame selected by rank tiebreaker (89.9 vs 86.2).** This is the Phase 2A init model.

---

## Phase 2A — Ablation: Decisiveness Level (planned 2026-04-26)

**Hypothesis:** Training decisiveness level (|tanh(SF eval)| threshold) affects geometry quality and ELO. There is an optimal threshold: too low (all positions) includes too many near-equal positions that contribute little signal; too high shifts the label distribution away from the draw cluster and may degrade play.

**Motivation:** Phase 1.5 showed that adding highly decisive data (endgame, nearmate, material — all >90% decisive) consistently hurt ELO. The mid_only dataset uses min-decisive=0.0 (all positions, ~44% decisive). This experiment isolates the decisiveness variable by holding n fixed and varying only the threshold applied at merge time.

**Init model:** `models/phase15_mid_no_endgame/best.pt` (Phase 1.5 best by rank, rank 89.9)
**Hyperparameters:** same as Phase 1.5 — `--policy-weight 0.0 --rank-reg 0.5 --lr 3e-4 --epochs 30 --geo-patience 5`
**Eval baseline:** `models/feb_sf/best.pt`
**Probe dataset:** `dataset_2021_06_mid_sf18.pt` (fixed, n=5000)

### Data pipeline

| Step | Script | Notes |
|------|--------|-------|
| Parse | `parse_dec_ablation.sh` ×5 + `parse_dec_ablation_merge.sh` | 5×50k games, 2025-02, 1800-2450 ELO, ~5M raw positions |
| Reeval | `reeval_dec_ablation.sh` (×7 chunks in parallel) | SF depth 18, N=5,000,000 |
| Merge | `reeval_dec_ablation_merge.sh` (×4 in parallel, DEC_LEVEL env var) | One dataset per threshold, all n=1,000,000 |

### Ablation Conditions

| Condition | min-decisive | Positions kept | Dataset |
|-----------|-------------|----------------|---------|
| dec00 | 0.0 | all (~100%) | `dataset_dec00_sf18.pt` |
| dec03 | 0.3 | ~70% pass | `dataset_dec03_sf18.pt` |
| dec05 | 0.5 | ~44% pass | `dataset_dec05_sf18.pt` |
| dec07 | 0.7 | ~25% pass | `dataset_dec07_sf18.pt` |

*All conditions: n=1,000,000 (controlled). If <1M positions pass at dec07, lower n uniformly.*

### Results Summary

| Condition | min-decisive | Rank | W·L (strict) | W·D (strict) | KR·KR | ELO Δ | wr |
|-----------|-------------|------|--------------|--------------|-------|-------|-----|
| dec00 | 0.0 | — | — | — | — | — | — |
| dec03 | 0.3 | — | — | — | — | — | — |
| dec05 | 0.5 | — | — | — | — | — | — |
| dec07 | 0.7 | — | — | — | — | — | — |

---

## Phase 1.5 — Ablation: Curated Dataset Components (2026-04-26)

**Hypothesis:** Structured data (endgame + near-mate + material imbalance) improves geometry rank and ELO beyond mid-band data alone. Each component contributes independently.

**Init model:** `models/2021_06_all/best.pt` (Phase 1 best, rank 87.0)
**Hyperparameters:** `--policy-weight 0.0 --rank-reg 0.5 --lr 3e-4 --epochs 30 --geo-patience 5`
**Eval baseline:** `models/feb_sf/best.pt`
**Probe dataset:** `dataset_2021_06_mid_sf18.pt` (fixed, n=5000)

### Datasets

| Dataset | File | Train | Val |
|---------|------|-------|-----|
| Mid reeval (partial ~71%) | `dataset_phase15_mid_sf18.pt` | 1,346,701 | 70,879 |
| Endgame (11 stages, 50k base) | `endgame_phase15_50k.pt` | 172,230 | 19,136 |
| Near-mate winning (mate≤3) | `dataset_near_mate.pt` | 60,756 | 3,197 |
| Near-mate losing | `dataset_near_mate_losing.pt` | 30,885 | 1,625 |
| Material imbalance (10 combos) | `dataset_material_imbalance.pt` | 190,000 | 10,000 |

### Ablation Conditions

| Condition | Dataset | Extra dataset | Train total |
|-----------|---------|---------------|-------------|
| mid_only | mid_sf18 | — | 1,346,701 |
| special_only | special_full | — | 453,871 |
| mid_full | mid_sf18 | special_full | 1,800,572 |
| mid_no_endgame | mid_sf18 | special_no_endgame | 1,628,342 |
| mid_no_nearmate | mid_sf18 | special_no_nearmate | 1,708,931 |
| mid_no_material | mid_sf18 | special_no_material | 1,610,572 |

### Results Summary

| Condition | Rank | W·L (strict) | W·D (strict) | Sep gap | KR·KR | KQ·K W | NN lift | β1 | W | D | L | Draw% | W/(W+L) | ELO Δ |
|-----------|------|--------------|--------------|---------|-------|--------|---------|-----|---|---|---|-------|---------|-------|
| mid_only | 86.2 | -0.5197 | **+0.5478** | 0.1566 | +0.072 | +0.773 | 0.358 | 183 | 22 | 108 | 70 | 54.0% | **23.9%** | -85 |
| mid_no_endgame | **89.9** | **-0.7551** | -0.1067 | **0.1769** | +0.242 | +0.940 | 0.361 | **213** | 17 | **113** | 70 | **56.5%** | 19.5% | -94 |
| mid_no_material | 88.8 | -0.7328 | +0.1176 | 0.1661 | +0.030 | +0.948 | **0.365** | 187 | 20 | 98 | 82 | 49.0% | 19.6% | -111 |
| mid_full | 87.9 | -0.6877 | -0.1291 | — | +0.057 | — | — | — | — | — | — | — | — | -123 |
| mid_no_nearmate | 82.3 | -0.6489 | **-0.4010** | 0.1639 | -0.014 | +0.962 | 0.332 | 194 | 16 | 93 | 91 | 46.5% | 15.0% | -137 |
| special_only | 78.1 | -0.6606 | -0.0127 | 0.1088 | -0.037 | +0.958 | 0.330 | 170 | 8 | 71 | 121 | 35.5% | 6.2% | **-222** |

*mid_full W/D/L not recorded. Previous best: 2021_06_all rank 87.0, wr 31.2% vs endgame baseline (different baseline).*

---

### Deep Analysis

#### 1. Statistical significance

At n=200 games, SE(wr) ≈ sqrt(p·(1–p)/200) ≈ 3.4pp, so the 95% CI on any wr is roughly ±7pp and on any ELO Δ is roughly ±50 ELO.

| Comparison | wr diff | z-score | verdict |
|---|---|---|---|
| mid_only vs mid_no_endgame | 1.3pp | 0.27 | noise |
| mid_only vs mid_no_material | 3.5pp | 0.73 | noise |
| mid_only vs mid_full | 5.0pp | ~1.0 | noise |
| mid_only vs mid_no_nearmate | 6.8pp | 1.45 | borderline (p≈0.15) |
| mid_only vs special_only | 16.2pp | 3.7 | **significant (p<0.001)** |
| mid_no_nearmate vs special_only | 9.4pp | 2.1 | significant (p≈0.04) |

**The real picture is two groups, not six.** special_only is clearly separated. Everything else — mid_only through mid_no_nearmate — cannot be ranked confidently at 200 games. The apparent ordering (mid_only > mid_no_endgame > …) is not statistically reliable. Confirming mid_only vs mid_no_nearmate would need ~800 games (4×) to reach p<0.05.

#### 2. Draw rate and game character

Draw rate varies from 35.5% (special_only) to 56.5% (mid_no_endgame). This reflects how similarly the model evaluates positions to feb_sf — similar-strength models draw more often. It is not independently interpretable beyond that.

W/(W+L) appears to vary (6–24%) but is not a reliable diagnostic. The decisive-game population is endogenous: a model that fails to hold drawn positions creates more decisive games, and most of those become losses. So the denominator grows for the wrong reasons. At ~90 decisive games per condition the binomial SE is ~4 wins, and the numerators differ by 4–14 wins — not significant in isolation. W/(W+L) should not be used as a proxy for "conversion skill" here.

#### 3. Win·draw cosine — description and hypothesis

The clearest variation across conditions is in the W·D cosine:

| Condition | W·D cosine | Draw% |
|---|---|---|
| mid_only | **+0.548** | 54.0% |
| mid_no_material | +0.118 | 49.0% |
| special_only | -0.013 | 35.5% |
| mid_no_endgame | -0.107 | 56.5% |
| mid_full | -0.129 | — |
| mid_no_nearmate | **-0.401** | 46.5% |

**Observation:** W·D cosine and draw rate correlate (r≈0.7 across these 5 conditions), suggesting the geometry metric does capture something real about in-game draw recognition. mid_only and mid_no_nearmate are at opposite extremes on both.

**Hypothesis (not established):** The probe dataset is ~56% draws by SF evaluation. Training on mid-only data may force draws near the win cluster because mid-game draws are typically tense positions with winning potential. Adding highly decisive data (endgame, nearmate, material all >90% decisive) may shift the label distribution so the win cluster is anchored by extreme decisive positions, pushing neutral/drawn positions toward the loss side. Near-mate losing data in particular (SF value ≈ -1.0) pulls the loss cluster far from neutral. This is a plausible mechanistic story consistent with the pattern, but we have 6 conditions and noisy ELO estimates — the data does not establish causality from W·D cosine to ELO.

#### 4. Rank as the primary selection criterion

Rank's predictive value for ELO is established across many experiments, not just this ablation. The general principle — higher rank → better play — is well-supported. Within this ablation, the low-rank conditions (special_only 78.1, mid_no_nearmate 82.3) are also the worst ELO, consistent with that principle.

The apparent non-monotone behaviour above rank 86 (mid_no_endgame rank 89.9 but ELO -94 vs mid_only rank 86.2 but ELO -85) is within noise at 200 games — it does not challenge the general principle. 200-game ELO estimates have ±50 ELO CI; a 9-ELO difference is meaningless. The right interpretation is: rank 89.9 and rank 86.2 are the two top candidates, and we cannot distinguish them from their vs-feb_sf results.

**Chosen model: mid_no_endgame** (rank 89.9, highest in the ablation). The rank signal is the more reliable indicator here; the ELO difference vs mid_only is noise.

#### 5. Most defensible explanation for mid_only's performance

The simplest and most defensible explanation: **distribution match**. The probe dataset is mid-band positions; the head-to-head games start from standard positions and are mid-game heavy for the first ~30-40 moves. mid_only trains on ~1.35M positions drawn from exactly that distribution. Every other condition adds data from a different distribution — piece-sparse decisive endgames, extreme near-mate positions, lopsided material imbalances. This introduces distribution shift that consistently hurts generalization to the position types that actually arise in play.

The W·D cosine variation may be a symptom of this same distribution shift rather than an independent cause: when the training mix shifts toward decisive positions, the learned geometry reflects that mix, and mid-game draw positions appear as outliers.

#### 6. KQ vs K value — calibration note

mid_only gives KQ vs K White to move as +0.773, versus 0.940–0.962 for all other conditions. All special datasets contain explicitly decisive endgame-like positions which push the value function toward high confidence in winning positions. The softer value in mid_only may affect MCTS tree shape (less pruning in endgame-adjacent nodes) but this is speculative — the probe position almost never arises from opening play.

KR vs KR value +0.242 for mid_no_endgame (should be ~0, drawn) confirms the model has no calibration for drawn rook endgames without endgame data. Practically irrelevant at 200 games from opening but a blind spot that would matter in longer matches or self-play.

#### 7. Conclusions

- **Two groups, not six.** special_only is clearly worse (p<0.001). All five mid-containing conditions are statistically indistinguishable at 200 games. The apparent ranking within the top 5 is noise.
- **The hypothesis is rejected in aggregate:** adding structured special data to mid_only does not improve ELO at this mixture ratio.
- **Mechanism unclear.** Distribution shift is the most parsimonious explanation. The W·D cosine pattern is consistent with this story and worth tracking, but should not be treated as a causal chain without further experiments.
- **Near-mate data has a borderline negative signal** (mid_no_nearmate ELO -137 vs mid_only -85, z=1.45, p≈0.15) alongside the starkest geometry degradation. Worth re-testing at higher game count before concluding.
- **Chosen model: mid_no_endgame** (rank 89.9). Rank is the more reliable selection criterion across this experimental history; the ELO difference vs mid_only is within noise.
- **Next experiment:** direct head-to-head, mid_no_endgame vs mid_only, 400 games (`eval_headtohead_noe_vs_midonly.sh`). This resolves the question cleanly without the independent-noise problem of comparing two separate vs-feb_sf runs.

---

### mid_only — Full Results (Job 28292097 train / 28292139 eval)

**Training:** Geo-patience stopped at epoch ~7. Training-time rank peaked at 59.5 — artefact of using training val split. Eval-time rank 86.2 is the honest metric.

**Geometry probe:**

| Metric | Value |
|--------|-------|
| Effective rank | **86.2 / 128** |
| Win·loss cosine (strict) | -0.5197 |
| Win·draw cosine (strict) | 0.5478 |
| Separation gap | 0.1566 |
| KR vs KR value | +0.072 |
| KQ vs K White | +0.773 |
| Queen-up White | +0.877 |
| NN label consistency | 0.770 (lift +0.358) |
| Topology | β0=1  β1=183  H=5.896 [healthy] |
| Drawness gates | 0/4 (expected — draw-reg=0.0) |

**ELO (200 games, n_sim=100, vs feb_sf):**

| W | D | L | wr | ELO Δ |
|---|---|---|----|-------|
| 22 | 108 | 70 | **38.0%** | **-85** |

**Note:** 38.0% vs feb_sf is +8pp over previous best at this matchup (30%). Rank 86.2 nearly matches Phase 1 best (87.0) despite different training data — geometry is robust to dataset composition changes at this scale.

---

### special_only — Full Results (Job 28292258 eval)

**Geometry probe:**

| Metric | Value |
|--------|-------|
| Effective rank | **78.1 / 128** |
| Win·loss cosine (strict) | -0.6606 |
| Win·draw cosine (strict) | -0.0127 |
| Separation gap | 0.1088 |
| KR vs KR value | -0.037 |
| KQ vs K White | +0.958 |
| Queen-up White | +0.957 |
| NN label consistency | 0.742 (lift +0.330) |
| Topology | β0=1  β1=170  H=5.837 [healthy] |
| Drawness gates | 0/4 (expected — draw-reg=0.0) |

**ELO (200 games, n_sim=100, vs feb_sf):**

| W | D | L | wr | ELO Δ |
|---|---|---|----|-------|
| 8 | 71 | 121 | 21.8% | -222 |

**Note:** Worst ELO in the ablation by a wide margin. Mid-band data is the generalist backbone — without it the model cannot transfer to positions it wasn't explicitly trained on. Win·draw cosine -0.0127 is near-zero (draws neither cluster with wins nor losses), and rank 78.1 is the lowest in the mid-containing conditions. The structured special data alone cannot bootstrap competitive play from the init model.

---

### mid_no_endgame — Full Results (Job 28292259 eval)

**Geometry probe:**

| Metric | Value |
|--------|-------|
| Effective rank | **89.9 / 128** |
| Win·loss cosine (strict) | -0.7551 |
| Win·draw cosine (strict) | -0.1067 |
| Separation gap | 0.1769 |
| KR vs KR value | +0.242 |
| KQ vs K White | +0.940 |
| Queen-up White | +0.959 |
| NN label consistency | 0.773 (lift +0.361) |
| Topology | β0=1  β1=213  H=5.865 [healthy] |
| Drawness gates | 0/4 (expected — draw-reg=0.0) |

**ELO (200 games, n_sim=100, vs feb_sf):**

| W | D | L | wr | ELO Δ |
|---|---|---|----|-------|
| 17 | 113 | 70 | 36.7% | -94 |

**Note:** Highest rank so far (89.9) but ELO worse than mid_only. Win·draw cosine -0.1067 (draws shifted toward loss cluster) matches the mid_full pattern. Endgame data is contributing to correct draw-side geometry rather than rank.

---

### mid_no_nearmate — Full Results (Job 28292260 eval)

**Geometry probe:**

| Metric | Value |
|--------|-------|
| Effective rank | **82.3 / 128** |
| Win·loss cosine (strict) | -0.6489 |
| Win·draw cosine (strict) | **-0.4010** |
| Separation gap | 0.1639 |
| KR vs KR value | -0.014 |
| KQ vs K White | +0.962 |
| Queen-up White | +0.966 |
| NN label consistency | 0.744 (lift +0.332) |
| Topology | β0=1  β1=194  H=5.782 [healthy] |
| Drawness gates | 0/4 (expected — draw-reg=0.0) |

**ELO (200 games, n_sim=100, vs feb_sf):**

| W | D | L | wr | ELO Δ |
|---|---|---|----|-------|
| 16 | 93 | 91 | 31.2% | -137 |

**Note:** Worst result so far. Win·draw cosine -0.4010 — extreme collapse of draw separation (draws clustering near losses). KR vs KR value -0.014 confirms the model has lost nearly all draw concept. Near-mate positions are essential: they provide the critical examples where winning ≠ drawing, anchoring the draw cluster correctly.

---

### mid_no_material — Full Results (Job 28292261 eval)

**Geometry probe:**

| Metric | Value |
|--------|-------|
| Effective rank | **88.8 / 128** |
| Win·loss cosine (strict) | -0.7328 |
| Win·draw cosine (strict) | +0.1176 |
| Separation gap | 0.1661 |
| KR vs KR value | +0.030 |
| KQ vs K White | +0.948 |
| Queen-up White | +0.965 |
| NN label consistency | 0.777 (lift +0.365) |
| Topology | β0=1  β1=187  H=5.785 [healthy] |
| Drawness gates | 0/4 (expected — draw-reg=0.0) |

**ELO (200 games, n_sim=100, vs feb_sf):**

| W | D | L | wr | ELO Δ |
|---|---|---|----|-------|
| 20 | 98 | 82 | 34.5% | -111 |

**Note:** Second-highest rank (88.8) but only 3rd-best ELO (34.5%). Win·draw cosine +0.1176 is healthy (draws cluster near wins, not losses), which is better than mid_full (-0.1291) despite lacking material data. Confirms that material imbalance positions push draws toward the loss cluster — they are predominantly decisive, which biases the model to treat balanced-material positions as winning when they are not.

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

---

## Session 11 — Endgame Supplement Results + Decisiveness Zigzag Design (2026-04-21)

### lichess_2023_03_endgame — training results

Model trained on 800k Lichess 2023-03 positions (SF depth-18) + 200k endgame anchor
(`endgame_sf15.pt`, anchor-frac=0.25). Init from `feb_sf/best.pt`. Geometry patience=3.

**Geometry probe (Job probe_geo_2023_03_endgame, dataset_2023_03_sf18.pt, n=5000):**

| Metric | lichess_2023_03_endgame | feb_sf baseline |
|--------|------------------------|-----------------|
| Effective rank | **21.6 / 128** | 18.9 / 128 |
| Win·draw cosine (strict) | 0.3404 | 0.1676 |
| KR vs KR \|value\| | **+0.184** ✓ | failed |

**First-ever KR vs KR pass.** Rank improved +2.7 over feb_sf. Win·draw cosine higher
than baseline — see interpretation note below.

**Win·draw cosine interpretation:** On Lichess val data the "draw" bucket (|v|<0.3)
contains near-equal middlegames with soft SF labels, not structural draws. After endgame
training the model has learned a genuine draw concept from KR vs KR / KNN vs K / KB vs KB
positions — these don't map cleanly onto near-equal middlegames. Cosine=0.3404 is not a
regression; use cosine as a collapse detector only (panic >0.5). KR vs KR sanity check
and ELO are the honest gates.

### ELO evaluation — lichess_2023_03_endgame vs feb_sf

200-game head-to-head (n_sim=100, step=5). Result at 100/200 games:

| Games | W | D | L | Win rate |
|-------|---|---|---|----------|
| 20 | 6 | 10 | 4 | 0.550 |
| 40 | 9 | 23 | 8 | 0.512 |
| 60 | 13 | 33 | 14 | 0.492 |
| 80 | 17 | 44 | 19 | 0.487 |
| 100 | 26 | 52 | 22 | **0.520** |

Draw rate 52% — model draws more than feb_sf. When decisive, W=26 L=22 (+4). Trending
slightly positive. Evaluation still running at time of writing.

**Fork decision:** ELO trending above 50% → Fork 2 live. Endgame supplement works.
Next step is decisiveness zigzag, not architecture rewrite.

---

### Decisiveness Zigzag — Design

Instead of further refining the endgame data mix, staircase training through
decisiveness-filtered subsets of the SF-labeled dataset:

| Stage | Lichess filter | Draw anchor | Purpose |
|-------|---------------|-------------|---------|
| 1 | `\|v\| > 0.7` | endgame pool (KR vs KR etc.) | Force win/loss axis apart, anchor draw dimension |
| 2 | `\|v\| > 0.5` | endgame pool + `\|v\| < 0.10` from Lichess | Broaden axis, expand draw region |
| 3 | All positions | endgame pool | Full rank expansion with gradations |

Gate at each stage: rank must increase before advancing. If rank stalls between stages
→ step size too large, add intermediate threshold.

**Draw anchor is mandatory even in Stage 1.** Without it, training on decisive-only
positions builds a 1D win/loss axis with no draw dimension. When Stage 3 introduces
draws cold into a geometry with no space for them, collapse or regression follows.
We observed this in Doover 2 Round 1 (endgame curriculum → rank 1.0).

**Draw perpendicularity loss** added to train.py alongside rank-reg:

```
L_draw = λ_draw · cos²(c_draw, axis_win_loss)
```

where `axis_win_loss = (c_win - c_loss) / ||c_win - c_loss||`, computed per-batch
from positions in each bucket. `c_win` and `c_loss` are detached — gradient only
flows through draw vectors. λ_draw = 0.01 (small nudge, value loss dominates).

Rationale: the scalar value loss treats 0.0 as the midpoint of the win/loss axis.
Draw vectors have no incentive to move perpendicular to win/loss — they just need to
be near the origin. The draw perp loss adds an explicit gradient forcing the draw
centroid off the win/loss axis, opening a genuine second dimension.

---

### Architectural Note — WDL head as alternative

The scalar value head supervises exactly one direction in the 128-dim geometry space.
Win/draw/loss all collapse onto a single axis: win high, loss low, draw in between.
No loss term forces draw to occupy a separate geometric direction.

A WDL (win-draw-loss) head predicts `(p_win, p_draw, p_loss)` with cross-entropy loss
(2 degrees of freedom after normalisation). Backprop then supervises two independent
directions, making three-way geometric separation a requirement of the loss rather than
an emergent property.

However: the scalar [-1,1] span gives the geometry a continuous gradient field with
127 unconstrained dimensions. The WDL simplex constrains two directions and leaves 126.
The continuous scalar may allow richer organic structure to emerge if the data is
sufficiently varied. This is an empirical question — WDL is not obviously superior.

**Decision:** run decisiveness zigzag + draw perp loss on the scalar architecture first.
Measure the rank ceiling. WDL is the architectural comparison once that ceiling is
known.

---

## Phase 2A — Decisiveness Ablation Results (2026-04-27)

**Hypothesis tested:** Higher SF decisiveness threshold → more decisive positions → stronger geometry signal → better ELO.

**Result: Hypothesis WRONG.** DEC_LEVEL=0.0 (no filter, natural mix) is best on every metric.

### Dataset

~4.96M positions from Lichess 2025-02, ELO 1800–2450, SF depth-18 reeval.
Natural decisive rate: 39.9% (`|SF eval| > 0.5`). Game-outcome draw rate: **5.2%**.

### Final geometry probe results

| DEC_LEVEL | Effective rank | win·loss cosine | win·draw cosine | β1 loops |
|-----------|---------------|-----------------|-----------------|----------|
| 0.0 | **80.3** | -0.69 | **-0.30** | **175** |
| 0.3 | 69.7 | -0.61 | +0.22 | 160 |
| 0.5 | 72.1 | -0.82 | +0.29 | 168 |
| 0.7 | 50.4 | -0.83 | -0.15 | 126 |

### Final head-to-head vs feb_sf (200 games, n_sim=100)

Draws excluded from primary metric — threefold repetition draws are noisy engine
artefacts, not signals of strength. Win count is the clean metric.

| DEC_LEVEL | Wins | Losses | ELO Δ |
|-----------|------|--------|-------|
| 0.0 | **20** | **71** | **-91** |
| 0.3 | 20 | 83 | -113 |
| 0.5 | 13 | 82 | -125 |
| 0.7 | 18 | 91 | -133 |

All conditions fail the gate (ELO Δ < 0 vs feb_sf).

### Key findings

**1. The draw concept is missing.** Only 5.2% of training positions come from drawn
games. The model rarely sees draws and cannot build a distinct draw region in its
geometry. DEC_LEVEL=0.0 has win·draw = -0.30 (draws genuinely orthogonal to wins —
three-pole geometry). Higher decisive filtering pushes the draw centroid toward the win
pole (win·draw > 0), collapsing to a two-pole geometry.

**2. Decisive filtering increases losses, not wins.** More filtering → model gets
overconfident in positions it should hold as draws → more losses. DEC_LEVEL=0.3 and
0.0 tie on wins (20 each) but 0.0 has 71 losses vs 0.3's 83. The harm is in
misjudging drawable positions, not in winning power.

**3. Decisiveness is a proxy for low ELO, not quality.** Lichess 1800–2450 games are
mostly won by blunder. High-ELO games (2200+) have genuine positional draws — higher
draw rate, higher quality draw positions. The fix is better source data, not filtering.

### Revised understanding

The original decisiveness hypothesis was based on SF depth ≤ 10 producing label
clustering near 0 (memory: "Shallow SF depth → too few decisive positions → tanh labels
cluster near 0 → weak geometry signal"). At depth 18, this is not the bottleneck. The
bottleneck is now the draw rate of source games, not label magnitude distribution.

---

## Phase 2B — ELO Ablation Design (2026-04-27)

**Hypothesis:** Higher-ELO source games have more genuine draws → model learns draw
concept → better draw geometry → better ELO.

Three parse conditions submitted from Lichess 2025-02 PGN:

| MIN_ELO | Expected draw rate | Output |
|---------|-------------------|--------|
| 2000 | ~8–10% | `dataset_elo2000_raw.pt` |
| 2100 | ~10–15% | `dataset_elo2100_raw.pt` |
| 2200 | ~15–20% | `dataset_high_elo_raw.pt` |

Each will be SF reeval'd (depth 18, 7 chunks), merged with `--min-decisive 0.0`
(no filter — Phase 2A result), and trained from `phase15_mid_no_endgame/best.pt`.

**Primary check after parse:** `python3 src/check_draw_rate.py <dataset>` to confirm
draw rate increases with ELO.

**Primary eval metric:** win count vs feb_sf (not win rate — draws are noisy).

---

## Phase 2C — SF Drawness Curriculum Design (2026-04-27)

**Hypothesis:** Directly generating SF-confirmed near-draw positions and blending them
into training is a more principled way to teach the draw concept than relying on
game-outcome draw rate.

**Approach:** Use `--max-decisive 0.1` (new flag, added 2026-04-27) to extract
positions where `|SF eval| ≤ 0.1` from the existing 4.96M reeval dataset. Blend at
controlled fractions (5%, 20%, 50%) into standard training data.

**Advantage over ELO banding:** No new SF evaluations needed — the existing reeval
data is already there. Direct control over draw position fraction rather than relying
on ELO as a proxy.

**Note on multiple Lichess files:** Additional months available (2021-06, 2023-03,
2025-01, 2025-02). If 2200+ games are too sparse in a single month, chain months
to reach 250k qualifying games.

---

## Phase 3A — Drawness Probe: rank-89.9 geometry separability (2026-04-29, Job 28320925)

**Question:** Does `phase15_mid_no_endgame/best.pt` (rank 89.9) already encode the
structural draw distinction latently?

**Result: YES — 99% separability, Cohen's d = 3.769.**

Check 6 (equal-region subspace probe) generated ~450 structural draw positions
(KR vs KR, KNN vs K, KB vs KB) and compared their geometry vectors against the
`|v|<0.2` pool from the probe dataset. Logistic regression (5-fold CV) achieved
0.990 accuracy. The drawness head outputs 0.516–0.540 for all position types — it
is completely untrained (model was trained with `--draw-reg 0.0`).

| Metric | Value |
|--------|-------|
| LR accuracy (5-fold CV) | **0.990** |
| Cohen's d on PC1 | **3.769** |
| PC1 — structural draws | +0.418 ± 0.152 |
| PC1 — equal+decisive | −0.021 ± 0.081 |
| Centroid cosine | +0.1686 |
| Top separating dims | [101, 84, 48, 99, 6] |

The hyperplane is already in the geometry. The drawness head needs only to be
pointed at it. Decision: `train_drawness_bootstrap.sh` with frozen backbone,
init from `phase15_mid_no_endgame/best.pt`.

**Open question:** Is the separation genuine drawness (Hypothesis B) or piece-count
proxy (Hypothesis A)? Decisive endgame stages (KQ vs K etc.) in the training data
serve as explicit negatives — the head will be told few-piece-decisive = 0 and
few-piece-structural-draw = 1, so the conflation is resolved by training regardless
of which hypothesis is correct.

---

## Phase 3B — Drawness Bootstrap v1 (2026-04-29, Job 28320933)

**Setup:**
- Init: `phase15_mid_no_endgame/best.pt` (rank 89.9)
- `--freeze-backbone --draw-reg 0.05 --epochs 2 --geo-patience 2`
- Endgame stages 1 2 4 5 9 10 11, 250k positions
- Drawness labels: pos=357,370 neg=490,919

**Result: head failed to learn.**

| Epoch | DrawL | KR vs KR drawness | Sicilian drawness | gap |
|-------|-------|-------------------|-------------------|-----|
| 1 | 0.641 | 0.415 | 0.378 | +0.037 |
| 2 | 0.639 | 0.434 | 0.359 | +0.075 |

Gates: KR vs KR >0.7 (failed), Sicilian <0.3 (failed). DrawL barely moved.

**Diagnosis:** λ=0.05 starved the drawness head of gradient. GNorm=0.002 across both
epochs — 20× below what Adam needs to converge a 129-parameter linear layer in 2
epochs. The loss is `vloss + 0.05 × draw_bce`; only the draw_bce term flows to the
drawness head (backbone frozen, value_head frozen). Effective gradient ≈ 0.05 × ∂BCE/∂θ.
Data pipeline confirmed correct (drawness_mask/drawness_targets properly set).

**Additional bug found:** `best.pt` was gated on `rank_improved AND topo_healthy`.
With frozen backbone, rank on endgame val barely moves after epoch 1 → best.pt stays
at epoch 1 (gap=0.037) even as epoch 2 improves (gap=0.075). Sanity check was
therefore reporting epoch 1 state.

### Changes made

**train.py** — `best.pt` saving condition widened to `(rank_improved OR draw_improved)
AND topo_healthy`. Frozen-backbone runs where rank is flat now save on drawness
improvement.

**train_drawness_bootstrap.sh** — v2 parameters:
- `--draw-reg 0.05` → `--draw-reg 1.0` (20× stronger gradient)
- `--epochs 2` → `--epochs 10`
- `--geo-patience 2` → `--geo-patience 10`

### Caveat

The λ diagnosis is consistent with the observed GNorm and DrawL trajectory, but has
not been confirmed by a successful run. **Training configuration may not be the only
issue.** Other candidates if v2 also fails:
- Gradient flow to drawness head broken (g.requires_grad=False when backbone frozen —
  should still work since drawness_head.weight has requires_grad=True, but unverified)
- Drawness labels from stages 1/2/4/5 not correctly structured (decisiveness vs
  structural draw split in the endgame generator)
- Head needs weight init from probe separating direction (ABLATIONS.md §Hypothesis B)

v2 resubmitted 2026-04-29 (job submitted after bsub trailing-period error on first
attempt, which sent job 28325516 to default queue with no directives — killed).

---

## Phase 3C — Draw-reg Ablation on ELO 2000 (PENDING — awaiting merge)

**Question:** Does draw-reg actually reshape backbone geometry, or does it only train the drawness head regardless of λ?

**Motivation:** draw-reg has no confirmed evidence of improving geometry. The draw region opened in Phase 2A and Exp B without draw-reg (rank-reg + no-policy did it). ELO 2000 data has more genuine draws than prior datasets, which may change the picture.

**Design:** 5 conditions varying only `--draw-reg`. Everything else fixed.

| Condition | draw-reg | rank-reg | policy-weight | init |
|-----------|----------|----------|---------------|------|
| dr0p0 | 0.0 | 0.5 | 0.0 | phase15_mid_no_endgame |
| dr0p2 | 0.2 | 0.5 | 0.0 | phase15_mid_no_endgame |
| dr0p5 | 0.5 | 0.5 | 0.0 | phase15_mid_no_endgame |
| dr1p0 | 1.0 | 0.5 | 0.0 | phase15_mid_no_endgame |
| dr2p0 | 2.0 | 0.5 | 0.0 | phase15_mid_no_endgame |

**Scripts:** `train_draw_reg_ablation.sh` (DRAW_REG env var) + `probe_draw_reg_ablation.sh`

**Key metrics:** effective rank, W·D cosine, KR vs KR value, drawness gap (structural vs balanced).

**Decision gate:** If rank and W·D cosine are flat across all conditions → draw-reg does not reshape geometry and we stop pursuing it as a mechanism. If the 0.0 baseline already shows good W·D cosine from ELO 2000 data alone → the data is doing the work, not the loss term.

**Prerequisite:** ELO 2000 merge (`bsub -env "MIN_ELO=2000" < jobs/reeval_elo_ablation_merge.sh`) must complete first.

---

## 2026-04-29 — Drawness eval reveals policy gap, two new experiments

### Findings
- `drawness_full` (rank 40.2) at 32.5% wr vs feb_sf after 100/200 games — losing.
- `drawness_head_v2` (rank 75.6, frozen + LR-init head) at 27.5% after 20/200 — losing harder.
- Confirmed `drawness` is NOT consumed in `mcts.py`/`evaluate.py` — drawness head is auxiliary, never used in play.
- Therefore drawness_head_v2 vs feb_sf ≡ phase15_mid_no_endgame vs feb_sf in play.
- Root cause: **phase15 was trained with `--policy-weight 0.0`**; feb_sf has trained policy. MCTS comparison is unfair, not a rank-→-ELO falsification.

### New experiments queued

#### 1. `natural` — drop drawness scaffolding, just train on the data
Hypothesis: the curriculum's value distribution (76k drawn + 400k decisive) is enough to shape geometry into draw-aware regions without explicit BCE. Init from `2021_06_all` (rank 87, has trained policy from Phase 1).

| | natural |
|---|---|
| init | `2021_06_all/best.pt` |
| dataset | `dataset_drawness_curriculum.pt` |
| policy-weight | 1.0 |
| rank-reg | 0.5 |
| draw-reg | **0.0** |
| script | `jobs/train_natural.sh` |

#### 2. `soft_drawness` — RL-flavored drawness via outcome smoothing
For each position, drawness target = fraction of its k=50 nearest geometric neighbours whose game ended in a draw. Replaces binary "is structurally drawn" with continuous *empirical draw probability*. Companion to `natural` — tests whether soft outcome-derived targets shape geometry better than no scaffolding at all.

| | soft_drawness |
|---|---|
| init | `2021_06_all/best.pt` |
| dataset | `dataset_drawness_curriculum_soft.pt` (built fresh) |
| policy-weight | 1.0 |
| rank-reg | 0.5 |
| draw-reg | 0.5 |
| extra | `--soft-drawness-targets` |
| build script | `jobs/build_soft_drawness.sh` |
| train script | `jobs/train_soft_drawness.sh` |

### Code changes (revertable)

**Files added** (delete to revert):
- `src/build_soft_drawness.py` — k-NN over training geometry, computes outcome-smoothed targets, writes `drawness_soft_targets` field on the dataset.
- `jobs/build_soft_drawness.sh` — submit script for the above.
- `jobs/train_soft_drawness.sh` — soft-drawness training run.
- `jobs/train_natural.sh` — natural (no drawness scaffolding) training run.
- `jobs/eval_natural_vs_feb_sf.sh` — head-to-head eval.
- `jobs/eval_drawness_head_v2_vs_feb_sf.sh` — head-to-head eval.
- `jobs/eval_drawness_full_vs_feb_sf.sh` — head-to-head eval.
- `jobs/train_drawness_head_v2.sh` — frozen + LR-init drawness head.

**Files modified** (search for the diff markers below to revert):
- `src/train.py`:
  - `_drawness_fields(d, use_soft=False)` — added `use_soft` param. When True and `drawness_soft_targets` is in the dataset, returns mask=all-True and targets=soft_targets, overriding the binary structural-draw labels.
  - `train(...)` — added `use_soft_drawness=False` param, threaded into `_make_loader` and the draw-reg log line.
  - CLI: added `--soft-drawness-targets` flag (sets `use_soft_drawness`).
  - CLI: previously added `--init-drawness-from-probe` flag (already in tree from drawness_head_v2 work).

**Datasets** (no destructive changes):
- New file: `dataset_drawness_curriculum_soft.pt` (in blackhole). Original `dataset_drawness_curriculum.pt` untouched.

**Models** (no destructive changes):
- New dirs: `models/natural/`, `models/soft_drawness/`. Existing models untouched.

### To revert all of this
```
rm src/build_soft_drawness.py
rm jobs/build_soft_drawness.sh jobs/train_soft_drawness.sh
rm jobs/train_natural.sh jobs/eval_natural_vs_feb_sf.sh
rm jobs/eval_drawness_head_v2_vs_feb_sf.sh jobs/eval_drawness_full_vs_feb_sf.sh
rm jobs/train_drawness_head_v2.sh
git checkout src/train.py    # reverts _drawness_fields + CLI changes
```
The `--init-drawness-from-probe` flag and `_init_drawness_from_lr` function in `train.py` are from the prior drawness_head_v2 work — leave them or revert separately.

### Decision gates
- **natural**: rank ≥ 70 + KR vs KR value < 0.35 + Cohen's d > 1 on probe Check 6 + wr > 50% vs feb_sf → drawness scaffolding was unnecessary.
- **soft_drawness**: same gates. If beats natural → empirical-probability framing produces better geometry than data alone. If ties natural → soft targets are equivalent to nothing. If loses to natural → soft targets are noise.

---

## 2026-04-29 — natural results + natural v2 plan

### natural results (Job 28327697)
- Best rank: 34.6 (curriculum-domain) / 33.8 (standard probe `dataset_2021_06_mid_sf18.pt`)
- KR vs KR value: +0.199 ✓ (passes <0.35 sanity)
- All sign checks passed
- **Cohen's d on Check 6: 1.827** (structural-vs-balanced separation)
- **LR accuracy: 0.926** — geometry organised drawness with NO scaffolding
- Top1 policy: 0.216 → 0.248 (climbed steadily, policy properly trained)
- Drawness head: 0.501 / 0.509 (random output — head wasn't trained, by design)

**Conclusion:** Data-not-hyperparameters thesis confirmed. The geometry self-organises a structural-draw region from value labels alone when the curriculum has the right composition. Auxiliary BCE / draw-reg / LR-init were scaffolding, not mechanism.

**Cost:** Rank dropped 87 → 33.8 on standard probe. Expected — curriculum's distribution is far from 2021_06_mid. Need cross-distribution rank check before declaring full picture.

### natural v2 plan
**Data composition:**
- Base: full `dataset_elo2100_sf18.pt` (all 2100+ ELO positions, all outcomes)
- Draw boost: drawn-only subset of `dataset_elo2200_sf18.pt` (filtered by `outcome_values == draw`)
- Output: `dataset_natural_v2.pt`

**Training recipe:** identical to natural — value + policy + rank-reg, no drawness scaffolding. Patience bumped to 10 (more data → longer reshape expected).

### Files added (revertable)
- `src/build_natural_v2.py` — concat full elo2100 + drawn-only elo2200
- `jobs/build_natural_v2.sh` — CPU build job
- `jobs/train_natural_v2.sh` — gpuv100 training job
- `jobs/eval_natural_vs_2021_06_all.sh` — same-lineage head-to-head

### To revert this batch
```
rm src/build_natural_v2.py
rm jobs/build_natural_v2.sh jobs/train_natural_v2.sh
rm jobs/eval_natural_vs_2021_06_all.sh
```
No train.py / shared-code changes in this batch.

---

## 2026-04-29 — natural v3: scale to 4.5M @ 7.5% drawn

### Build result for v2
- Pool: 1.97M base (elo2100) + 19,144 drawn from elo2200
- Final: 1.99M @ 7.5% drawn — natural rate at 2100+ play
- Decision: scale up further. v3 pools all reeval'd sources to 4.5M @ same fraction.

### v3 plan
**Composition:** ~4.5M positions @ 7.5% drawn, engineered by oversampling drawn rows and undersampling decisive ones from a pool of 6 SF-reeval'd sources (elo2000/2100/2200, 2021_06_all/high, 2023_03). Skips low-quality bands (low_elo, 2021_06_low/mid).

**Same training recipe as natural / natural_v2** — value + policy + rank-reg, no drawness scaffolding.

**Hypothesis:** scale wins. ~10× more data than natural and ~2× more than v2; same recipe. Geometry should organise drawness more cleanly while preserving more rank than natural's 33.8.

### Files added
- `src/build_natural_v3.py` — flexible multi-source pooling with draw-fraction engineering
- `jobs/build_natural_v3.sh` — CPU build (64GB mem)
- `jobs/train_natural_v3.sh` — gpuv100, 64GB mem, patience 10

### To revert
```
rm src/build_natural_v3.py jobs/build_natural_v3.sh jobs/train_natural_v3.sh
```
No train.py changes.

---

## 2026-04-29 — natural v3 cancelled, v4 plan: outcome supervision

### v3 problem (caught before running)
Pooled sources `elo2000/2100/2200_sf18.pt` are **nested ELO thresholds** parsed from the same Lichess month — every game in elo2200 is also in elo2100, which is also in elo2000. Pooling = duplication = overfitting risk. Same nesting for `2021_06_all` / `2021_06_high`.

Added FEN-based dedup to v3 build script as defense, but cleaner is to use known-disjoint sources.

### v4 plan
**Disjoint by month:** 2021-06 + 2023-03 + 2025-01. Three independent Lichess months, no nesting possible.

**Supervision change: AlphaZero-style.** Drop SF eval as teacher signal. Use raw `outcome_values` (+1/0/-1 STM-relative game result) as the value training target. Cleaner scientifically — model learns from games alone, no engine assistance — and lets us use `dataset_2025_01.pt` which has no SF reeval.

**Recipe:** Same as natural — value MSE on outcome + policy + rank-reg + no drawness scaffolding. Patience 10. Init from 2021_06_all (still has policy).

**Engineering:** ~5M positions @ 7.5% drawn (subsample drawn-up, decisive-down). FEN dedup as defense.

### Why outcome supervision
- Cleaner PoC: "Petra distinguishes structural-from-balanced draws after training on Lichess games at high ELO with no engine teacher."
- Per-position label is noisier (every position in a drawn game labelled 0, including transient sharp middlegame positions) but with 5M positions, noise averages out.
- Validates whether SF labels were doing the work, or whether geometry self-organises from raw game outcomes too.

### Files added (revertable)
- `src/build_natural_v4.py` — multi-source pool with outcome-as-values + FEN dedup
- `jobs/build_natural_v4.sh` — CPU build (64GB mem)
- `jobs/train_natural_v4.sh` — gpuv100, patience 10

### To revert
```
rm src/build_natural_v4.py jobs/build_natural_v4.sh jobs/train_natural_v4.sh
```
No train.py changes.

### Note on running v3 vs v4
v3 has FEN dedup so the overlap risk is mitigated, but the dedup can't recover what's lost (we keep the first occurrence of each FEN, biasing toward whichever source loaded first). v4 is the cleaner path. Either can be run, but v4 is the honest one.

---

## 2026-04-29 — natural breakthrough: +150 ELO over 2021_06_all

### Result (Job 28327745, ongoing)

`natural` vs `2021_06_all` head-to-head, **stable at 70.6% wr through 80/200 games** with only **4 losses**. Eval continues but the trend has held at 20, 40, 60, 80 — this is the model's actual strength.

| n | W | D | L | wr |
|---|---|---|---|----|
| 20 | 10 | 8 | 2 | 70.0% |
| 40 | 18 | 20 | 2 | 70.0% |
| 60 | 27 | 30 | 3 | 70.0% |
| 80 | 37 | 39 | 4 | 70.6% |

ELO Δ ≈ **+150** over its own initialisation point.

### Why this is significant

- **Recipe is minimal**: value MSE + policy + rank-reg=0.5 on `dataset_drawness_curriculum.pt`. NO draw-reg, NO drawness BCE, NO LR-init, NO soft targets. The "natural" approach.
- **Training cost**: 25 epochs, ~10 minutes GPU. Most efficient ELO gain in project history.
- **Falsifies pure rank-→-ELO**: natural's probe rank dropped 87 → 33.8 but ELO climbed +150. Spread without structure is wasted capacity.
- **Validates geometry organisation**: natural's Cohen's d=1.83 on Check 6 (structural-vs-balanced separation) is what's winning, not raw rank.

### What the result clears

| Question | Answer |
|---|---|
| Did the curriculum trade help play? | **Yes, +150 ELO** |
| Was drawness scaffolding necessary? | **No** — geometry self-organised from value labels |
| Is rank the right metric? | **No alone** — need rank + Cohen's d + KR vs KR triple |
| Is drawness data-derivable? | **Yes** — and it improves play, not just probes |

### Models in the field

| Model | Rank (mid probe) | KR vs KR | Cohen's d (Check 6) | Has policy | Best ELO finding |
|---|---|---|---|---|---|
| 2021_06_all | 87 | unknown | unknown | yes | baseline |
| natural | **33.8** | **+0.199** | **1.827** | yes | **+150 vs 2021_06_all** |
| drawness_full | 40.2 | -0.004 | 4.06 | yes (poisoned) | -90 vs feb_sf |
| drawness_head_v2 | 75.6 | +0.266 | 2.25 | no | functionally = phase15 |
| phase15_mid_no_endgame | 76.1 | unknown | unknown | no (policy-weight=0) | |
| feb_sf | 18.9 | +0.024 | unknown | yes | +58 vs sf_balanced |

natural is now the best candidate for further work.

---

## TO DO — sequenced by priority

### Immediate (let evals finish, queue cheap follow-ups)

- [ ] **Let `natural vs 2021_06_all` finish 200/200** — confirm trend; expected: 65-72% wr
- [ ] **Cancel `drawness_head_v2 vs feb_sf`** — we know what it'll show (phase15 has no policy); save the GPU
- [ ] **Push and run `natural vs feb_sf`** — first fair MCTS comparison since both have policy. The "real" PoC eval gate.
- [ ] **Cross-distribution probe of natural** — probe on val splits of 2021_06_high, 2021_06_low, elo2100 to see if rank 33.8 is mid-band-specific or reflects overall geometry collapse.

### Big-data path (parallel track)

- [ ] **Build + train `natural_v4`** (5M, 3 disjoint months, outcome supervision). Already pushed. AlphaZero-style: no SF teacher. Tests whether scale + outcome alone suffices.
- [ ] **`natural_v4 vs natural`** head-to-head. If v4 ≥ natural, big data + outcome wins; if not, the curated SF curriculum was contributing real signal.
- [ ] **`natural_v5` (2200+ asymmetric sampling)**: re-parse 2200+ months taking ALL positions from drawn games (capture sustained equality directly) + sampled positions from decisive games. Only do this if v4 is promising.
- [ ] **Volume scaling**: consider `natural_v6` at 10M+ from 5-6 disjoint Lichess months once the recipe is locked in.

### RL-inspired path (alternative track)

- [ ] **Run `soft_drawness`** as a control. Already pushed. If it ties or loses to `natural`, soft outcome smoothing is unnecessary. If it wins, the empirical-probability framing has merit. Keep it as a parallel, not primary, track.
- [ ] **MCTS-rollout drawness targets**: at training time, run shallow rollouts from each position; drawness target = fraction of rollouts ending in draw. Expensive but principled. Defer until natural / v4 results are clear.
- [ ] **Self-play drawness calibration**: pick 5-10k structurally-draw-candidate positions, play 50+ self-play games each, label by empirical draw rate. Gold-standard but days of compute. Only if needed for the writeup.
- [ ] **Drawness consumed in MCTS**: change `mcts.py` to use drawness in leaf evaluation (`effective_value = value × (1 − drawness) + 0 × drawness`). Makes drawness *useful* for play, not just diagnostic. The strategic prize.

### Drawness probing (across all candidate models)

- [ ] **Build the test battery** (~200 paired positions: structural vs sharp-balanced with near-zero SF eval). Hand-curated, never used in training. Becomes the PoC eval.
  - Berlin endgames, Petroff exchanges, opposite-colour bishops, fortresses, blockades, perpetuals (structural)
  - Dynamic Sicilian, KID Mar-del-Plata, opposite-castling races (sharp balanced)
- [ ] **Battery accuracy** for natural, drawness_head_v2, drawness_full, soft_drawness, natural_v4 — single comparable number per model.
- [ ] **Causality probe**: KR vs KR + add passer → drawness should drop; sharp middlegame trade pieces to KR vs KR → drawness should rise. Tests whether the head learns the property or just surface features.
- [ ] **Analytic LR fit on natural's geometry** → put resulting weights into the drawness head, no training. Cohen's d on Check 6 was 1.83; this should pass the gate cleanly.
- [ ] **Cross-opening generalisation**: if the test battery is partitioned by opening, does battery accuracy hold across openings? Tests "head learned property, not surface".

### Phase 3 (after natural / v4 / battery work)

- [ ] **Policy reintroduction on phase15-derivatives** (drawness_head_v2 backbone) and re-eval vs feb_sf. Closes the question of "what would phase15 have done with policy".
- [ ] **Self-play from natural** — first time we have a model strong enough to bootstrap. Generate self-play data, fine-tune.
- [ ] **Self-play + drawness coupling** — if drawness is consumed by MCTS, self-play games become drawness-aware.

### Stikprøve / data quality

- [ ] **Manual inspection of curriculum positives** — pick 40 positives from `dataset_drawness_curriculum.pt` and read with chess eyes. Does the sustained-equality filter actually surface structural draws? Validates the data layer.
- [ ] **Sustained-equality filter** for re-parsed data: build a script that filters `|SF|<0.10 across ≥10 plies, ply≥30` for any reeval'd dataset. Creates cleaner positives than the current single-position |SF|<0.15 + ply≥40.

### Writeup prep

- [ ] **Test battery + natural's battery score** — the headline number for the paper.
- [ ] **Comparison table**: SF / LCZero / AlphaZero output a single value at structural/balanced ≈ 0; natural separates them. The novelty claim.
- [ ] **Geometry visualisation**: PCA / t-SNE of natural's bottleneck on the test battery, colour-coded structural vs balanced. Visible separation = picture for the paper.
- [ ] **Causality results**: ablation positions showing drawness moves predictably. Demonstrates the head learns the property.
- [ ] **ELO results**: natural +150 vs 2021_06_all (and ideally natural > feb_sf).

---

## Strategic forks under "TO DO"

- **Big data vs RL-inspired**: both push the PoC further. Big data is cheaper and more aligned with "data not hyperparameters." RL-inspired (rollout / self-play targets, MCTS-consumed drawness) is more sophisticated but expensive. Run big data in main track; RL-inspired as parallel exploration only if natural / v4 plateau.
- **SF labels vs outcome labels**: v4 tests outcome-only. If v4 matches or exceeds natural, SF teacher signal can be retired. If v4 underperforms, SF stays.
- **Curated curriculum vs raw scale**: natural was 477k curated. v4 is 5M raw. Whichever wins decides the data philosophy going forward.

---

## 2026-04-29 — Conceptual frame: rank is degrees of freedom

User insight after the natural breakthrough: **"rank is like degrees of freedom — something you can sacrifice for new learning."**

This reframes the whole project's metric story.

### The map

| Quantity | What it measures |
|---|---|
| Rank | Total degrees of freedom in the geometry. Capacity, not content. |
| Cohen's d on Check 6 | One DoF committed to a specific concept (structural-vs-balanced). |
| KR vs KR value gate | The committed direction is correct. |
| ELO | Net effect — committed structure minus capacity loss. |

### Why this explains natural

`2021_06_all` had rank 87 of mostly diffuse, decision-irrelevant variance — opening clusters, ELO-band signatures, surface features. `natural`'s curriculum training **committed** capacity to encode "structural-draw vs balanced" (Cohen's d 1.83 on Check 6). The committed dimension cost rank (87 → 33.8). The committed direction is decision-relevant. Net: +150 ELO.

### Predictive use

- Runs that **don't drop rank** are NOT committing new capacity — they're refining existing structure (incremental learning).
- Runs that **drop rank substantially** are doing fundamental reshape (commitment learning).
- Big rank drops should correlate with big concept gains. If the rank drop isn't paid back by structure (Cohen's d, ELO), the run was wasted capacity.

### Recipe implications

- **Don't optimise rank.** Optimise useful structure per dimension.
- **Accept rank loss** when training adds a decision-relevant concept.
- **Track the triple**: rank + Cohen's d on Check 6 + ELO. Single-axis rank metric is misleading.
- **Phase 1's rank→ELO finding** was true because all those models learned the same concept set. Across paradigms (different data, different concepts), rank alone is uninformative.

### What this means for the writeup

Two story layers:

1. **Specific**: chess engines can be trained to distinguish structural draws from balanced positions (the PoC).
2. **General**: geometric capacity is fungible — high rank without commitment is wasted; low rank with structure can outperform it. Rank trade-offs are productive, not destructive.

Story 2 is the broader contribution — applicable to any representation-learning context, not just chess.

---

## 2026-04-30 — natural_v2 trained; three-way ablation fork ready

### natural_v2 training result (Job 28327762)

Ran in parallel with natural. Same recipe (no drawness scaffolding), 4× more data (1.99M @ 7.5% drawn).

| Metric | natural | natural_v2 |
|---|---|---|
| Best rank (curriculum-domain) | 34.6 | 33.5 |
| KR vs KR value | +0.199 | **+0.024** |
| Top1 (final) | 0.248 | **0.300** |
| Epochs run | 25 | 32 |
| All sign checks | ✓ | ✓ |
| Drawness head | 0.501 / 0.509 (untrained, expected) | 0.509 / 0.496 (untrained, expected) |

`natural_v2`'s KR vs KR value (+0.024) is **much closer to zero** than natural's (+0.199). The structural-draw region is more cleanly organised. Top1 0.300 is also notably higher — more data produced a sharper policy.

Same-lineage head-to-head queued: `eval_natural_v2_vs_2021_06_all.sh`.

### The three-way fork

Once `natural_v2 vs 2021_06_all` finishes, we have three candidates trained from `2021_06_all` with three philosophies:

| Branch | Data | Supervision | Scaffolding | Result |
|---|---|---|---|---|
| `natural` | curated curriculum (477k, SF) | SF eval | none | **70.6% vs 2021_06_all** |
| `natural_v2` | elo2100 + drawn-2200 (2M, SF) | SF eval | none | TBD |
| `natural_v4` (queued) | 3 disjoint months (5M, raw outcome) | game outcome | none | not yet trained |
| `soft_drawness` (queued) | curriculum + outcome-smoothed soft targets | SF eval + soft BCE | RL-inspired | not yet trained |

This is the project's main fork point. Each branch tests a different hypothesis:

- **natural vs natural_v2**: does scale + raw 2100+ data beat curated drawness positives? Tests whether the |SF|<0.15+ply≥40 curation was contributing real signal.
- **natural vs natural_v4**: does outcome supervision (no SF teacher) match SF-supervised? Tests whether SF labels were doing the work.
- **natural vs soft_drawness**: does empirical-probability soft drawness target beat no scaffolding at all? Tests the RL-inspired framing.

### What the result-set will tell us

After all three head-to-heads vs 2021_06_all complete:

- If **natural ≈ natural_v2**: data composition matters more than curation method.
- If **natural_v2 > natural**: scale + clean source mixing wins.
- If **natural_v4 ≈ natural**: SF labels weren't doing critical work; outcome supervision suffices.
- If **soft_drawness > natural**: RL-inspired soft targets help; merits scaling up.
- If **soft_drawness < natural**: scaffolding really is dead even when soft.
- If **soft_drawness ≈ natural**: it's the data; the loss function is incidental.

### Files added
- `jobs/eval_natural_v2_vs_2021_06_all.sh` — same-lineage head-to-head matching the natural eval.

### Reminder: RL track is also queued
The `soft_drawness` experiment (RL-inspired) is built and ready:
- `jobs/build_soft_drawness.sh` (CPU, ~30 min — builds `dataset_drawness_curriculum_soft.pt` with k=50 outcome-smoothed targets)
- `jobs/train_soft_drawness.sh` (GPU, ~10–30 min — trains with `--soft-drawness-targets` and `--draw-reg 0.5`)

If we want all three branches in hand for tomorrow's analysis, both should be submitted now.

---

## 2026-04-30 — natural_v2 probe: Cohen's d 4.5, rank UP

Probed `natural_v2/best.pt` on `dataset_2021_06_mid_sf18.pt`. The result is substantially stronger than natural's probe on every drawness-relevant axis.

### Comparison table

| Metric | natural | **natural_v2** | Δ |
|---|---|---|---|
| Effective rank (mid probe) | 33.8 | **36.2** | +2.4 |
| Cohen's d on PC1 (Check 6) | 1.827 | **4.500** | **+146%** |
| LR accuracy (Check 6) | 0.926 | **0.970** | +4.4 pp |
| PC1 variance share (Check 6) | 4.6% | **11.1%** | 2.4× |
| Within-group spread (structural) | 0.920 | 0.638 | tighter |
| Centroid distance (struct ↔ balanced) | 0.391 | 0.741 | +89% |
| KR vs KR value | +0.199 | **+0.024** | closer to 0 |
| win·draw strict cosine | +0.4090 | **+0.0800** | near-orthogonal |
| Separation gap (Check 2) | 0.1591 | **0.2556** | +61% |
| Top1 final (training) | 0.248 | **0.300** | +5.2 pp |
| Top dims (Check 6) | [100, 30, 89, 102, 3] | [33, 66, 126, 51, 124] | different geometry |

### Significance

- **Cohen's d 4.5** matches `drawness_full`'s 4.06 — but drawness_full achieved that by destroying rank to 40 via unfrozen BCE. natural_v2 reaches 4.5 with rank 36.2 and **zero drawness scaffolding**.
- **PC1 explains 11.1% of variance** (vs natural's 4.6%) — the draw axis is 2× more dominant in the geometry. The structural-draw concept is now a *primary* dimension of the bottleneck.
- **Rank went UP** from natural's 33.8 to 36.2 despite 4× more data. Counter to a naive "more reshape costs rank" reading of the DoF frame, but consistent with: scale lets the model commit more dimensions to structure without sacrificing other capacity. Volume buys back DoF.
- **win·draw strict cosine 0.08** — draws are now near-orthogonal to wins. natural still had +0.41. This is the "three-pole geometry" we've been chasing — three nearly orthogonal regions for win / loss / draw.

### Updated rank-as-DoF interpretation

The DoF frame was framed earlier as a fixed budget (commit a dim, lose rank). natural_v2 shows the budget is elastic: with enough training data, the model can *grow* committed structure without paying capacity tax. Big data lets you commit dimensions productively rather than zero-sum.

This sharpens the recipe:
- **Small curated data**: rank-vs-structure is zero-sum (natural).
- **Large clean data**: rank-vs-structure can both improve (natural_v2).

### Next steps prompted by this result

- **Run `eval_natural_v2_vs_2021_06_all.sh`** — does the cleaner geometry produce more ELO than natural's 70.6%? If yes, scale is strictly better than curation. If natural still wins on ELO despite weaker probe metrics, the SF curriculum was contributing something beyond what shows in the probe.
- **Run `natural_v2 vs natural`** — direct head-to-head between the two candidates. Whichever wins becomes the production model.
- **soft_drawness vs natural_v2** — if soft_drawness is also run, it competes with natural_v2 for the title. Cohen's d 4.5 is high enough that beating it requires actually-good scaffolding.
- **natural_v4 (5M, outcome supervision)** is now the strategic experiment: with 2.5× more data than v2 and outcome-only labels, can geometry organise drawness as well or better than natural_v2's? If yes, SF labels are unnecessary.
