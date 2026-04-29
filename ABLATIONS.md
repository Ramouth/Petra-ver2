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
| 3C | Draw-reg weight (λ_draw) | TBD | round 1 invalid (drawness label bug); round 2 pending |

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

### Probe results (Job 28320925, 2026-04-29)

**Check 6 — Equal-region subspace:**

| Metric | Value |
|--------|-------|
| Logistic regression accuracy (5-fold CV) | **0.990** |
| Cohen's d on PC1 | **3.769** |
| PC1 projection — structural draws | +0.418 ± 0.152 |
| PC1 projection — equal+decisive | −0.021 ± 0.081 |
| Centroid cosine (structural vs equal+decisive) | +0.1686 |
| Top-5 separating dims | [101, 84, 48, 99, 6] |

**Verdict: SEPARABLE.** The geometry at rank 89.9 already separates structural draws from balanced positions with near-perfect accuracy. This is Hypothesis B confirmed at the extreme.

**Check 4 — Drawness head:**

All four probes score 0.516–0.540 — essentially constant regardless of position type. The drawness head is not failing; it is **completely untrained**. The model was trained with `--draw-reg 0.0`. The head has random initialisation and has never seen a gradient. It outputs ~0.5 for everything.

**Combined reading:** The geometry has the drawness distinction encoded. The head has never been asked to read it out. The separation is already there in the backbone — particularly along dimensions [101, 84, 48, 99, 6] and strongly on PC1 of the equal-region subspace (Cohen's d = 3.769, which is massive — anything above 1.0 is considered well-separated).

**One open question:** Is the separation piece-count driven (Hypothesis A) or genuine drawness (Hypothesis B)? KQ vs K (decisive, 3 pieces) and KR vs KR (structural draw, 4 pieces) differ in both drawness and piece count. The probe did not output projections for individual known positions onto PC1. A follow-up check — projecting KQ vs K onto the separating direction — would resolve this. If KQ vs K projects negative (same side as balanced positions), the separation is genuine drawness. If it projects positive (same side as structural draws), it is piece-count driven.

Practically: even under Hypothesis A, the frozen-backbone training with explicit negatives (decisive endgame stages 1–8 as drawness=0) will resolve the piece-count conflation automatically — the head will be told KQ vs K = 0 and KR vs KR = 1 explicitly.

**Next step:** `train_drawness_bootstrap.sh` with init from `phase15_mid_no_endgame/best.pt`. Given 99% linear separability the frozen head will converge in ≤2 epochs.

---

### Probe hypothesis and predicted train.py consequences

**Probe:** `probe_draw_subspace.sh` on `phase15_mid_no_endgame/best.pt` (rank 89.9).
Check 6 generates ~450 structural draw positions (KR vs KR, KNN vs K, KB vs KB) and compares their geometry vectors against the `|v|<0.2` pool from the probe dataset.

---

**Hypothesis A — High accuracy (>85%), piece-count driven** *(most likely)*

The CNN backbone encodes piece count directly (14-channel tensor includes piece presence per square). A 4-piece KR vs KR board looks completely different from a 16-piece Sicilian in the input tensor. The geometry likely separates them well — but for the wrong reason: it is separating *few-piece endgames* from *complex middlegames*, not *structural draws* from *sharp-balanced positions*.

Evidence this is what's happening: KQ vs K (decisive, 3 pieces) would also get high drawness from a frozen head trained this way. The PC1 separating direction would correlate with piece count, not draw concept.

**Consequence for train.py:** The drawness head needs explicit negatives that are also few-piece but decisive. Decisive endgames (stages 1–8 in endgame_drawness.pt: KQ vs K, KR vs K etc.) already exist in the anchor with drawness=0. The fix is to increase their frequency in the drawness batch so the head sees: few-piece decisive (target=0) AND few-piece structural draw (target=1) AND many-piece balanced (target=0). No new data needed — just anchor mix adjustment.

---

**Hypothesis B — High accuracy (>85%), genuine drawness** *(possible)*

The W·D cosine of −0.1067 at rank 89.9 means the draw cluster is already partially orthogonal to the win/loss axis. Structural draws (oscillatory, no-progress dynamics) may cluster in a distinct region from balanced middlegame positions even among the `|v|<0.2` pool. The logistic regression finds a real drawness hyperplane.

Evidence this is what's happening: KQ vs K (decisive few-piece) and KR vs KR (structural draw few-piece) would project to opposite sides of the separating direction.

**Consequence for train.py:** The backbone is already correct. Only the drawness head needs to find this hyperplane. Minimal fix: initialize `model.drawness_head.weight` to the logistic regression coefficient vector from the probe. Then freeze backbone, train 2 epochs on the existing anchor. The head starts already aligned with the correct direction instead of random init.

```python
# New utility: init drawness head from probe separating direction
w_probe = np.load("probe_draw_axis.npy")        # saved by probe
model.drawness_head.weight.data = torch.tensor(w_probe).unsqueeze(0)
```

---

**Hypothesis C — Low accuracy (~50%)** *(possible if W·D cosine −0.1067 is insufficient)*

The rank-89.9 geometry maps structural draws and sharp balanced positions to the same geometric region. The draw cluster exists globally (W·D cosine −0.1067) but the `|v|<0.2` subspace is not internally differentiated — structural draws and sharp balanced are neighbours.

Evidence this is what's happening: centroid cosine between group A and group B close to 1.0, Cohen's d on PC1 < 0.3.

**Consequence for train.py:** The backbone must change. The minimal targeted fix is extending the `draw_neg` mask in the BCE loss to include balanced-but-decisive positions — filling the supervision gap directly.

Currently (train.py line 466):
```python
draw_neg = values.abs() > 0.5
```

Proposed change — add `outcome_values` to the batch and extend the negative mask:
```python
draw_neg_decisive  = values.abs() > 0.5
draw_neg_balanced  = (outcome_values.abs() > 0.5) & (values.abs() < 0.2)
draw_neg           = draw_neg_decisive | draw_neg_balanced
```

Where `outcome_values` is the game outcome (±1 for decisive, 0 for draw) stored in the dataset but currently not passed to the training loop at all. This requires:
1. Add `outcome_values` to the collated batch in `train.py` (currently absent from the batch entirely — only used in data filtering)
2. Change the `draw_neg` mask as above
3. Run with unfrozen backbone so the geometry can adapt (the gap has been there throughout all prior training)

This is the most invasive path but also the most principled — it directly supervises the distinction between "equal-looking but game was decisive" (balanced) and "equal-looking and game was drawn" (candidate structural draw).

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

---

## Phase 3C — Draw Regularisation Weight Ablation (2026-04-29)

**Question:** Does the draw-reg loss weight (λ_draw) reshape backbone geometry, or does it only train the drawness head in isolation?

**Hypothesis:** Higher λ_draw → more gradient through the backbone via the BCE drawness loss → draw cluster opens / rank changes. If rank and W·D cosine are flat across conditions, draw-reg is not a geometry mechanism and should be abandoned as a path.

**Init model:** `models/phase15_mid_no_endgame/best.pt` (rank 89.9)
**Dataset:** `dataset_elo2000_sf18.pt` (train: 1,463,000 / val: 77,000)
**Hyperparameters:** `--policy-weight 0.0 --rank-reg 0.5 --lr 3e-4 --epochs 20 --geo-patience 5 --num-workers 0`

### Conditions

| DRAW_REG | Model path |
|----------|-----------|
| 0.0 | `models/draw_reg_abl/dr0p0/best.pt` |
| 0.2 | `models/draw_reg_abl/dr0p2/best.pt` |
| 0.5 | `models/draw_reg_abl/dr0p5/best.pt` |
| 1.0 | `models/draw_reg_abl/dr1p0/best.pt` |
| 2.0 | `models/draw_reg_abl/dr2p0/best.pt` |

### Round 1 — INVALID (drawness label bug)

All 5 conditions trained on `dataset_elo2000_sf18.pt` with a critical drawness labelling error (see below). Results for conditions 0.2–2.0 are pending but will not be interpreted — the drawness signal was near-zero for all conditions, making the ablation a test of nothing.

**Drawness label bug (discovered 2026-04-29):**

The merge script used `--drawness-game-level --drawness-sf-threshold 0.11`. This requires the SF eval to NEVER exceed 0.11 tanh units (≈44 centipawns) for ANY sampled position in a drawn game. This inverts the correct null hypothesis.

The correct framing: a drawn ELO 2000 game is presumptively drawn. Each position in that game is a drawness positive unless SF sees real imbalance at that specific moment (|SF| ≥ threshold). Tactical fluctuations in other parts of the game are irrelevant to whether this position is drawn.

Result of the bug: only **2,186 drawness positives from 1,540,000 positions (0.14%)**. The draw-reg loss was essentially silent for 99.86% of training steps. No condition could show any effect.

**Fix:** Remove `--drawness-game-level`. Use position-level with `--drawness-sf-threshold 0.22` (≈90cp). Criterion: game outcome = draw AND |SF eval at this position| < 0.22. Expected yield: ~50–100k positives.

### DRAW_REG=0.0 — Probe Results (Job 28327364, 2026-04-29)

Trained 7 epochs (geometry patience exhausted). Probed on `dataset_2021_06_mid_sf18.pt` (n=5000).

| Metric | Value |
|--------|-------|
| Effective rank | **78.9** / 128 |
| win·draw cosine (strict) | −0.124 |
| win·loss cosine (strict) | −0.572 |
| loss·draw cosine (strict) | +0.409 |
| Separation gap | 0.187 |
| NN label lift | +0.363 |
| Check 6 LR accuracy | 0.990 |
| Check 6 Cohen's d (PC1) | **4.575** |
| Top separating dims | [101, 48, 80, 10, 84] |
| Topology β1 | 191 |
| Drawness gates passed | 0 / 4 (head untrained) |
| KR vs KR value | +0.149 |

**Key observations:**

1. **Rank 78.9 is healthy** — 11 points below the phase15_mid_no_endgame init (89.9), but well above the 30-point threshold. The ELO 2000 data partially relaxed the geometry but did not damage it.

2. **Check 6 Cohen's d improved: 3.769 → 4.575.** The structural draw / balanced position separation strengthened after 7 epochs of ELO 2000 training with zero drawness supervision. The higher-quality game data appears to sharpen the equal-region structure. Separating dims [101, 48, 84] are stable across init and trained models — these are the core draw-encoding axes.

3. **win·draw cosine −0.124** is marginally more negative than the init (−0.107). Draws are slightly more separated from wins after ELO 2000 training. The loss·draw collinearity (+0.409) persists — draws are still geometrically closer to losses than to wins, which is structurally expected from the STM-relative value framing.

4. **Drawness head: all outputs ≈ 0.52** (random init). With DRAW_REG=0.0, the head received zero gradient. This confirms that Check 6 separability (99%) is purely a backbone property — the information is there but the head cannot read it without supervision.

5. **KNN vs K value = +0.851** — the model reads material (two knights vs one king) as winning, which is geometrically correct for a material-heavy position but wrong for a known fortress draw. This is the exact failure mode the drawness head should correct.

### Round 2 — Pending

After conditions 0.2–2.0 complete (for completeness), re-merge with corrected labels and retrain all 5 conditions. The Round 2 ablation will be the valid experiment.

**Round 2 submit commands (after re-merge):**
```bash
bsub -env "MIN_ELO=2000" < jobs/reeval_elo_ablation_merge.sh   # re-merge with fixed labels
# then retrain:
bsub -env "DRAW_REG=0.0" < jobs/train_draw_reg_ablation.sh
bsub -env "DRAW_REG=0.2" < jobs/train_draw_reg_ablation.sh
bsub -env "DRAW_REG=0.5" < jobs/train_draw_reg_ablation.sh
bsub -env "DRAW_REG=1.0" < jobs/train_draw_reg_ablation.sh
bsub -env "DRAW_REG=2.0" < jobs/train_draw_reg_ablation.sh
```

### Results Table (Round 1 — invalid)

| DRAW_REG | Epochs | Rank (probe) | W·D cos | Cohen's d | Drawness gates | ELO Δ vs phase15_noe |
|----------|--------|-------------|---------|-----------|----------------|----------------------|
| 0.0 | 7 | 78.9 | −0.124 | 4.575 | 0/4 | pending |
| 0.2 | — | — | — | — | — | — |
| 0.5 | — | — | — | — | — | — |
| 1.0 | — | — | — | — | — | — |
| 2.0 | — | — | — | — | — | — |
