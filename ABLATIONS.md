# Petra — Ablation Experiments

---

## Summary Table

| Phase | Variable | Winner | Key Finding |
|-------|----------|--------|-------------|
| 1.5 | Dataset components (mid + specials) | mid_no_endgame | special data hurts; mid is the backbone |
| 2A | Decisiveness filter threshold | dec00 (no filter) | filtering removes draws → geometry collapses |
| 2B | Source ELO band | TBD | higher ELO → more genuine draws (hypothesis) |
| 2C | SF draw blend fraction | TBD | direct draw injection vs ELO proxy |
| 3 (draw dim) | Draw dimension opening | TBD | see section below |

---

## Phase 3 — Opening the Draw Dimension

### The core problem

The model currently conflates two geometrically distinct concepts:

| Position | value q | drawness d | What we want |
|----------|---------|------------|--------------|
| KR vs KR (structural draw) | ≈ 0 | high | d → 1.0 |
| Sharp balanced middlegame | ≈ 0 | low | d → 0.0 |
| Decisive winning position | high | low | d → 0.0 |

Both KR vs KR and a sharp balanced Sicilian have `|SF eval| ≈ 0`. The model cannot distinguish them from scalar labels alone. "Equal" and "drawn" are not the same concept — drawn means *neither side can make progress*; equal means *both sides have chances*.

**Decisive gate:** `KR vs KR drawness > 0.7` AND `sharp balanced drawness < 0.3`. Until both pass simultaneously, the draw dimension is not solved.

---

### Signals that worked (from experiment history)

**1. Structural endgame anchors → KR vs KR value pass (lichess_2023_03_endgame)**

First-ever KR vs KR value near-zero pass: +0.184. Training on KR vs KR / KNN vs K / KB vs KB draw positions as explicit anchors taught the value head that these positions are balanced. W·D cosine 0.3404 — draws not collapsed with wins. Rank 21.6.

Mechanism: the endgame pool anchor (stages 9/10/11, drawness=1.0 explicit labels) gives the geometry a genuine draw cluster. The value signal alone (|v|≈0 on structural draws vs |v|≈0 on balanced midgames) is insufficient — structural draw examples are essential.

**2. Auxiliary drawness head partially learned (lichess_2023_03_drawness)**

Results: KNN vs K drawness 0.761 ✓ (passed), KR vs KR 0.658 (failed, gate >0.7), sharp balanced 0.410 (failed, gate <0.3). The head can learn — it is not the right architecture to separate structural draws from equal positions at this draw-reg / anchor mixture.

Rank regressed 21.6 → 19.5. The draw-reg BCE loss backprops through geometry even at small weight (0.05), pulling the space toward the endgame anchor subspace (low diversity → lower rank).

**3. Frozen backbone bootstrap achieves drawness gap = +0.956 in 1 epoch**

Freezing the backbone and training only the 129-parameter drawness head on explicit endgame labels converged almost instantly. Rank was preserved (21.6 guaranteed by frozen backbone). However, when backbone was then unfrozen for broad training, rank stayed at ~21 — the Phase 2A rank of 80.3 was not achievable from this lower-rank start.

Key lesson: the bootstrap/broaden sequence requires a high-rank base to broaden *from*. Starting from rank 21.6 limits the ceiling.

**4. Exp B (no policy, λ=0.5) opened the draw region at rank 72.9 — but played badly**

`train_exp_b_gpu.sh`: init from lichess_2023_03_endgame, no policy, rank λ=0.5, data = dataset_2023_03_sf18.pt + dataset_2025_01_low_elo_sf18.pt. Result: rank **72.9**.

**ELO result (Job 28265356, 200 games vs lichess_2023_03_endgame rank 21.6):**

| W | D | L | wr | ELO Δ |
|---|---|---|----|-------|
| 15 | 78 | 107 | 27.0% | **−173** |

Rank 72.9 with no policy loses to rank 21.6 with policy by 173 ELO. The draw region is geometrically open but the model cannot play chess without policy guidance. MCTS without a policy prior uniform-random-walks the tree — it cannot efficiently direct search toward good moves. The high rank is irrelevant if search cannot exploit it.

Draw rate 39% (78/200) is lower than typical (50–55% between equal models) and the W/L split (15/107) is extremely lopsided. The model draws by repetition when MCTS finds no direction, and loses all decisive games.

**Why the draw region opened (mechanism):**

Equal positions (`|v| ≈ 0`) produce near-zero gradient on the MSE value loss. The only signal moving them is rank regularization. At λ=0.5 + no policy, there is no competing gradient — all 128 dimensions are available for the rank pressure to spread equal positions outward into their own cluster, orthogonal to the win/loss axis. The low-ELO extra dataset (22.5% equal positions) gave that cluster enough mass to be geometrically coherent.

**Critical finding:** The conditions that open the draw region (no policy) are incompatible with competitive play. The draw region opens when policy competition is removed, but policy is necessary for MCTS to function. This is not a coincidence — both effects trace to the same cause: without policy, the geometry is entirely shaped by value and rank, which is better for draw geometry but makes search degenerate.

**5. Phase 2A dec00 — same mechanism, worse ELO relative to its baseline**

Rank 80.3, W·D cosine −0.30 (three-pole geometry). Same no-policy conditions. ELO −91 vs feb_sf (which is weaker than lichess_2023_03_endgame) — so Phase 2A is also a strong geometric success and a competitive failure for the same reason.

**Conclusion: the draw region opens under conditions that break play.**
- No policy + strong rank pressure → draw region opens (rank 70+)
- No policy → MCTS degenerates → ELO collapses regardless of geometry quality
- Policy + weak rank pressure → draw region stays closed (rank 21.6)

The real question is: **can the draw region open with policy ON?** That is what Exp C tested.

---

### Root cause of sharp-balanced failure

The drawness BCE loss (train.py:466) defines:
```python
draw_neg = values.abs() > 0.5    # decisive positions → target 0.0
draw_mask = drawness_mask | draw_neg
```

This creates a supervision gap: positions with `0.0 ≤ |SF eval| ≤ 0.5` receive **no gradient**. The sharp balanced Sicilian probe sits at `|v|≈0` — not an anchor (no positive label), not decisive (no negative label). The drawness head is never told it should be low. It interpolates from geometry proximity alone and settles at 0.41.

This is not an architecture problem. It is not a rank problem. It is a missing negative problem. The fix is to put balanced-but-decisive positions into the loss with target=0.0.

The ground truth for "balanced but not drawn" already exists in the dataset: `all_outcome_values` stores the game outcome separately from the SF position evaluation. A position where `|SF eval| < 0.2` AND `|game_outcome| > 0.5` is "equal-looking but the game proved it wasn't a draw." These are the natural negatives that are currently missing from the loss.

---

### Experiment 1 — Freeze-and-probe Phase 2A (DONE — signal confirmed latent)

**Question:** Does Phase 2A's geometry (rank 80.3, W·D −0.30) already encode the structural draw distinction latently, such that a frozen backbone + fresh drawness head can recover it?

**Result: YES** — with a caveat. The draw region is geometrically open and the drawness head can read it out. But the geometry was produced without a policy head, so this proves the draw region *can exist* in geometry space, not that we can get there from a model that also plays well.

**Setup:**
- Init: `models/dec00_sf18/best.pt` (Phase 2A winner, rank 80.3)
- Freeze backbone, train only drawness head on `endgame_drawness.pt`
- `--freeze-backbone --draw-reg 0.05 --epochs 2 --policy-weight 0.0`

**Why this worked:** At rank 80.3, the equal-position cluster is geometrically distinct from the win/loss axis (W·D cosine −0.30). The structural draw positions (KR vs KR, KNN vs K) map into that cluster. The drawness head — a single linear layer — was sufficient to separate them from sharp-balanced positions that sit in a different part of that cluster.

**The remaining problem (from Exp B ELO result):** The Phase 2A geometry was trained without policy. The no-policy model it sits on top of is ELO −91 vs feb_sf, and Exp B (same conditions, same rank regime) is −173 vs lichess_2023_03_endgame. The draw region is open in the geometry of a model that cannot play competitively. The key open question is now: **does the draw region also open in a model trained with policy?** That is what Exp C measures.

---

### Experiment 2 — Outcome-based negatives: freeze backbone, fix the supervision gap

**Question:** Does the rank-21.6 geometry (lichess_2023_03_endgame) already separate KR vs KR from sharp balanced — we just need to tell the drawness head what sharp balanced should score?

**What changes:** Add explicit drawness=0.0 negatives for balanced-but-decisive positions. The signal already exists in `all_outcome_values` in the dataset. Mine positions where `|SF eval| < 0.2` AND `|game_outcome| > 0.5` → drawness_target=0.0.

**Setup:**
- Init: `models/lichess_2023_03_endgame/best.pt` (rank 21.6, policy on, competitive)
- Freeze backbone (no rank regression possible)
- Anchor: `endgame_drawness.pt` — structural draw positives (KR vs KR etc., target=1.0)
- Mined negatives: from `dataset_2023_03_sf18.pt` + `dataset_2025_01_low_elo_sf18.pt`, positions where `|v|<0.2` AND `|outcome|>0.5` → target=0.0
- `--draw-reg 0.05 --freeze-backbone --epochs 3 --policy-weight 0.0`

**Ratio target:** ~1:1 positives to negatives. The structural draw anchor has ~50k positions (stages 9/10/11). Mine ~50k balanced-decisive positions from the main dataset.

**What this tests:** Is the rank-21.6 geometry sufficient, or does the draw region need to be open (rank 70+) before the head can separate them?

**If this fails:** The backbone geometry maps KR vs KR and sharp balanced to the same region. No linear head can separate them. Need rank ≥ 70 with policy — which is the unsolved problem.

**Gates:**
- KR vs KR drawness > 0.7
- Sharp balanced drawness < 0.3
- KNN vs K drawness > 0.7
- KQ vs K decisive drawness < 0.3
- Rank = 21.6 (unchanged, backbone frozen)

---

### Experiment 3 — SF WDL as continuous drawness label (ground truth signal)

**Question:** Can Stockfish's native WDL output provide the correct drawness signal across the full position distribution, enabling the head to generalise far beyond the endgame anchor?

**Why this is the best possible signal:**
- `|SF eval| ≈ 0` applies to both KR vs KR (p_draw≈0.95) and sharp balanced Sicilian (p_draw≈0.10). SF knows they are different.
- SF WDL captures dynamic draw probability: repetition risk, fortress likelihood, piece-count parity. It is not a threshold on the scalar — it is the engine's own draw concept.
- Yields ~30–40% of positions with clear drawness labels, vs ~5% from game-outcome alone.

**Implementation (reeval_stockfish.py):**
1. Add `setoption name UCI_ShowWDL value true` to SF initialisation
2. Parse `score wdl <w> <d> <l>` from SF output alongside cp/mate
3. Store `drawness_target = d/1000.0` and `drawness_mask = True` where `d/1000 > 0.6` (structural draw) or `d/1000 < 0.2` (not drawn); skip ambiguous middle
4. Re-evaluate the existing raw parsed dataset with the updated script

**Training:** Frozen backbone of lichess_2023_03_endgame, `--draw-reg 0.05 --freeze-backbone --epochs 3`. No anchor dataset needed — the WDL labels cover the full position distribution including structural draw endgames.

**Gates:** Same as Experiment 2.

---

### Experiment 4 — Pairwise contrastive loss (no new data, no new reeval)

**Question:** Can a margin ranking loss train the relative ordering — structural draws score higher than balanced-decisive — without needing absolute label values?

**When to use:** In parallel with Exp 2 as a cross-check; or as fallback if Exp 2's mined negatives are too sparse.

**Implementation:**
```python
# Per batch: pair each anchor draw row with a balanced-decisive main-dataset row
margin = 0.4
contrastive_loss = mean(relu(margin - d_structural + d_balanced_decisive))
```

Where `d_structural` = drawness head output on anchor positions (drawness=1.0), `d_balanced_decisive` = drawness head output on main-dataset positions with `|v|<0.2` AND `|outcome|>0.5`.

No new dataset field needed. The existing `is_anchor` flag and `values`/`outcome_values` tensors are sufficient to partition the batch.

**Advantage:** Trains the distinction directly rather than the absolute values. The head only needs to output "higher for structural draws than for balanced-decisive positions" — not to hit specific threshold values. More robust when geometry representations are similar.

**Gates:** Same as Experiment 2.

---

### Execution order

```
Root cause identified: supervision gap (0 < |v| < 0.5 gets no gradient)

Exp 2 (outcome-based negatives, frozen backbone)
  → PASS: rank-21.6 geometry already separates them → done, deploy
  → FAIL: geometry doesn't separate at rank 21.6 → need higher rank with policy

     ↓ if Exp 2 fails
Exp C (policy on, λ=0.5) result  ←── pending
  → If rank ≥ 60: freeze backbone, repeat Exp 2 on higher-rank geometry
  → If rank stays ~21: policy is ceiling, fundamentally different approach needed

Exp 3 (SF WDL) — run in parallel with Exp 2, independent of outcome
  → Best long-term signal regardless of which path resolves the problem

Exp 4 (contrastive) — run in parallel with Exp 2 as cross-check
```

---

### Results

| Condition | Policy | λ | Rank | W·D cosine | ELO (vs baseline) | Notes |
|-----------|--------|---|------|------------|-------------------|-------|
| lichess_2023_03_endgame | ✓ | 0.1 | 21.6 | 0.34 | baseline | best competitive model |
| Exp A (no policy, λ=0.1) | ✗ | 0.1 | 40.5 | — | — | rank improved, ELO unknown |
| Exp B (no policy, λ=0.5) | ✗ | 0.5 | 72.9 | — | **−173** vs endgame | draw region open, unplayable |
| Phase 2A dec00 | ✗ | 0.5 | 80.3 | −0.30 | −91 vs feb_sf | draw region open, unplayable |
| Exp C (policy, λ=0.5) | ✓ | 0.5 | — | — | — | pending |
| **Exp 2** (outcome negatives, frozen) | ✓ | — | 21.6 (frozen) | — | — | **next to run** |
| Exp 3 (SF WDL) | ✓ | — | — | — | — | requires reeval change |
| Exp 4 (contrastive) | ✓ | — | — | — | — | parallel with Exp 2 |

---

## Phase 1.5 — Dataset Components (2026-04-26)

**Question:** Does adding structured special data (endgame, near-mate, material imbalance) to mid-band improve geometry rank and ELO?

**Init model:** `models/2021_06_all/best.pt` (rank 87.0)
**Hyperparameters:** `--policy-weight 0.0 --rank-reg 0.5 --lr 3e-4 --epochs 30 --geo-patience 5`
**Eval baseline:** `models/feb_sf/best.pt`
**Probe dataset:** `dataset_2021_06_mid_sf18.pt` (n=5000, fixed)

### Datasets

| Dataset | File | Train | Val |
|---------|------|-------|-----|
| Mid reeval (~71%) | `dataset_phase15_mid_sf18.pt` | 1,346,701 | 70,879 |
| Endgame (11 stages, 50k base) | `endgame_phase15_50k.pt` | 172,230 | 19,136 |
| Near-mate winning (mate≤3) | `dataset_near_mate.pt` | 60,756 | 3,197 |
| Near-mate losing | `dataset_near_mate_losing.pt` | 30,885 | 1,625 |
| Material imbalance (10 combos) | `dataset_material_imbalance.pt` | 190,000 | 10,000 |

### Conditions

| Condition | Dataset | Extra dataset | Train total |
|-----------|---------|---------------|-------------|
| mid_only | mid_sf18 | — | 1,346,701 |
| special_only | special_full | — | 453,871 |
| mid_full | mid_sf18 | special_full | 1,800,572 |
| mid_no_endgame | mid_sf18 | special_no_endgame | 1,628,342 |
| mid_no_nearmate | mid_sf18 | special_no_nearmate | 1,708,931 |
| mid_no_material | mid_sf18 | special_no_material | 1,610,572 |

### Results

| Condition | Rank | W·L cosine | W·D cosine | Sep gap | KR·KR | ELO Δ | W | D | L | Draw% |
|-----------|------|------------|------------|---------|-------|-------|---|---|---|-------|
| mid_only | 86.2 | -0.5197 | **+0.5478** | 0.1566 | +0.072 | -85 | 22 | 108 | 70 | 54.0% |
| mid_no_endgame | **89.9** | **-0.7551** | -0.1067 | **0.1769** | +0.242 | -94 | 17 | 113 | 70 | 56.5% |
| mid_no_material | 88.8 | -0.7328 | +0.1176 | 0.1661 | +0.030 | -111 | 20 | 98 | 82 | 49.0% |
| mid_full | 87.9 | -0.6877 | -0.1291 | — | +0.057 | -123 | — | — | — | — |
| mid_no_nearmate | 82.3 | -0.6489 | **-0.4010** | 0.1639 | -0.014 | -137 | 16 | 93 | 91 | 46.5% |
| special_only | 78.1 | -0.6606 | -0.0127 | 0.1088 | -0.037 | **-222** | 8 | 71 | 121 | 35.5% |

### Statistical Significance (n=200, SE(wr)≈3.4pp, ±7pp 95% CI)

| Comparison | wr diff | z | verdict |
|---|---|---|---|
| mid_only vs mid_no_endgame | 1.3pp | 0.27 | noise |
| mid_only vs mid_no_material | 3.5pp | 0.73 | noise |
| mid_only vs mid_full | 5.0pp | ~1.0 | noise |
| mid_only vs mid_no_nearmate | 6.8pp | 1.45 | borderline (p≈0.15) |
| mid_only vs special_only | 16.2pp | 3.7 | **significant (p<0.001)** |

### Conclusions

- **Two groups, not six.** special_only is clearly worse. The five mid-containing conditions are statistically indistinguishable at 200 games.
- **Hypothesis rejected.** Adding structured special data to mid_only does not improve ELO at this mixture ratio.
- **Mechanism: distribution shift.** The probe dataset is mid-band heavy; special data shifts the training distribution, hurting generalization to the position types that actually arise in play.
- **Near-mate data has a borderline negative signal** (z=1.45, worst W·D cosine at -0.4010). Essential for draw concept — its removal collapses draw geometry.
- **Winner: mid_no_endgame** (rank 89.9, highest). The rank difference vs mid_only (89.9 vs 86.2) is the tiebreaker; the 9-ELO difference is noise.

### Follow-up: Head-to-head Confirmation (Job 28294106, 2026-04-26)

Direct 400-game match, mid_no_endgame vs mid_only (n_sim=100):

| W | D | L | wr | ELO Δ | 95% CI |
|---|---|---|----|-------|--------|
| 55 | 294 | 51 | 50.5% | +3 | [45.6%, 55.4%] |

z=0.2. Statistically identical. **mid_no_endgame selected by rank tiebreaker (89.9 vs 86.2).** This is the Phase 2A init model.

---

## Phase 2A — Decisiveness Filter (2026-04-27)

**Question:** Does filtering training data to include only decisive positions (high |SF eval|) improve geometry and ELO?

**Hypothesis:** More decisive positions → stronger label contrast → better geometry signal.

**Init model:** `models/phase15_mid_no_endgame/best.pt` (rank 89.9)
**Hyperparameters:** same as Phase 1.5
**Source data:** ~4.96M positions, Lichess 2025-02, ELO 1800–2450, SF depth-18
**Natural decisive rate:** 39.9% (`|SF eval| > 0.5`). Game-outcome draw rate: **5.2%**.

### Conditions

| Condition | min-decisive | Positions kept | Dataset |
|-----------|-------------|----------------|---------|
| dec00 | 0.0 | ~100% | `dataset_dec00_sf18.pt` |
| dec03 | 0.3 | ~70% | `dataset_dec03_sf18.pt` |
| dec05 | 0.5 | ~44% | `dataset_dec05_sf18.pt` |
| dec07 | 0.7 | ~25% | `dataset_dec07_sf18.pt` |

All conditions: n=1,000,000 (controlled).

### Results

**Geometry probe:**

| DEC_LEVEL | Effective rank | win·loss cosine | win·draw cosine | β1 loops |
|-----------|---------------|-----------------|-----------------|----------|
| **0.0** | **80.3** | -0.69 | **-0.30** | **175** |
| 0.3 | 69.7 | -0.61 | +0.22 | 160 |
| 0.5 | 72.1 | -0.82 | +0.29 | 168 |
| 0.7 | 50.4 | -0.83 | -0.15 | 126 |

**Head-to-head vs feb_sf (200 games, n_sim=100):**

| DEC_LEVEL | Wins | Losses | ELO Δ |
|-----------|------|--------|-------|
| **0.0** | **20** | **71** | **-91** |
| 0.3 | 20 | 83 | -113 |
| 0.5 | 13 | 82 | -125 |
| 0.7 | 18 | 91 | -133 |

All conditions fail the gate (ELO Δ < 0 vs feb_sf). Primary metric is win count — threefold draws are noisy engine artefacts.

### Key Findings

1. **Hypothesis WRONG.** dec00 (no filter) is best on every metric.

2. **The draw concept is missing.** Only 5.2% of training positions come from drawn games. The model rarely sees draws and cannot build a distinct draw region. dec00 has win·draw = -0.30 (draws genuinely orthogonal to wins — three-pole geometry). Higher filtering pushes the draw centroid toward the win pole (win·draw > 0), collapsing to a two-pole geometry.

3. **Decisive filtering increases losses, not wins.** More filtering → model gets overconfident in positions it should hold as draws → more losses. dec03 and dec00 tie on wins (20 each) but dec00 has 71 losses vs dec03's 83. The harm is in misjudging drawable positions.

4. **Decisiveness is a proxy for low ELO, not quality.** Lichess 1800–2450 games are mostly won by blunder. Higher-ELO games have genuine positional draws. The fix is better source data, not filtering.

### Revised Understanding

The original decisiveness hypothesis (memory: "Shallow SF depth → too few decisive positions → tanh labels cluster near 0 → weak geometry signal") was validated at SF depth ≤ 10. At depth 18, label magnitude is not the bottleneck. The bottleneck is the **draw rate of source games** — the geometry needs to see genuine draws to build a draw dimension.

---

## Phase 2B — Source ELO Band (planned 2026-04-27)

**Question:** Does training on higher-ELO games produce more genuine draws and thus better draw geometry?

**Hypothesis:** Higher-ELO → more positional draws → model builds genuine draw concept → better ELO.

**Init model:** `models/phase15_mid_no_endgame/best.pt`
**Filter:** `--min-decisive 0.0` (Phase 2A result — no filter)
**Source:** Lichess 2025-02

### Planned Conditions

| MIN_ELO | Expected draw rate | Dataset |
|---------|-------------------|---------|
| 2000 | ~8–10% | `dataset_elo2000_raw.pt` |
| 2100 | ~10–15% | `dataset_elo2100_raw.pt` |
| 2200 | ~15–20% | `dataset_high_elo_raw.pt` |

**Primary check after parse:** `python3 src/check_draw_rate.py <dataset>` to confirm draw rate increases with ELO.

**Primary eval metric:** win count vs feb_sf (not win rate).

### Results

| MIN_ELO | Draw rate (observed) | Rank | ELO Δ | Wins | Losses |
|---------|---------------------|------|-------|------|--------|
| 2000 | — | — | — | — | — |
| 2100 | — | — | — | — | — |
| 2200 | — | — | — | — | — |

---

## Phase 2C — SF Drawness Curriculum (planned 2026-04-27)

**Question:** Does directly blending SF-confirmed near-draw positions into training improve draw geometry?

**Hypothesis:** Direct draw injection is a more principled approach than relying on ELO as a proxy for draw rate.

**Approach:** Use `--max-decisive 0.1` to extract positions where `|SF eval| ≤ 0.1` from the existing 4.96M reeval dataset. Blend at controlled fractions into standard training data. No new SF evaluations needed.

**Init model:** `models/phase15_mid_no_endgame/best.pt`

### Planned Conditions

| Draw blend % | Training mix | Dataset |
|-------------|-------------|---------|
| 5% | 950k standard + 50k near-draw | TBD |
| 20% | 800k standard + 200k near-draw | TBD |
| 50% | 500k standard + 500k near-draw | TBD |

### Results

| Draw blend % | Rank | win·draw cosine | ELO Δ | Wins | Losses |
|-------------|------|-----------------|-------|------|--------|
| 5% | — | — | — | — | — |
| 20% | — | — | — | — | — |
| 50% | — | — | — | — | — |
