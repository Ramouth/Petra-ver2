# Debugging Log — Petra SF Model (Session Apr 10 2026)

## Context

Model: `models/best.pt` — supervised pretraining on Lichess data, value labels
re-evaluated with Stockfish depth 15 (200k positions, `dataset_sf.pt`).
Best val loss: 4.8011 (epoch 11/16, early stopped).

---

## Results So Far

### Step 2 — Greedy vs Random
```
Games: 100  W=0 D=100 L=0  wr=50.0%  ELO Δ=0
```
**100% draws.** Greedy policy cannot convert against random play in 300 half-moves.

### Step 5 — MCTS(learned) vs MCTS(material)
```
Games: 100  W=48 D=25 L=27  wr=60.5%  ELO Δ=+74
```
**GATE PASSED.** Learned value head beats material baseline. Self-play is greenlit
in principle, but we are debugging first.

### Step 6 — MCTS(geometry probe) vs MCTS(material)
```
Run 1: Games 100  W=2  D=8  L=90   wr=6.0%   ELO Δ=-478
Run 2: Games 100  W=0  D=3  L=97   wr=1.5%   ELO Δ=-727
```
Two runs confirm: geometry probe is **actively harmful**, not neutral. Variance
between runs (-478 vs -727) is expected at 100 games with random opening
temperature — not a meaningful difference.

PGN analysis (run 2, 100 games):
- 97 CHECKMATE, 3 move_limit — games terminate properly, not move-limit dominated
- Ply distribution: 40–300, median ~70 — real game lengths
- Material wins equally with both colors (48 as White, 49 as Black)
- Geometry lost all 97 decisive games regardless of color

The wins (0–2 across runs) are not signal — with probe outputs near 0, the
geometry player degenerates to policy-only search and can stumble into mates
by accident.

PGN review also reveals: MCTS(material) with two queens + rook took until move
73 to deliver checkmate (moves 33–44 show rook shuffling, king oscillating).
**Conversion is a problem for material evaluation too.** This is a policy head
issue — see Issue 1 below.

The probe direction in MCTS evaluation space is slightly wrong, making things
worse than random. R²=0.18 confirms real signal exists in the geometry, but
it needs self-play to strengthen and align. **Step 6 closed — needs self-play.**

### Geometry Tests — `test_geometry.py`
```
Probe R²  train=0.1824  val=0.1800
Test 1  FAIL  Material monotonicity  (Knight > Bishop breaks strict order)
Test 2  PASS  Piece value ordering   (Q > R > N > B > P, all pairwise checks pass)
Test 3  PASS  STM symmetry           (cosine 1.0 — correct, guaranteed by STM encoding)
Test 4  WARN  Forced mate convergence (2/4 non-monotone steps)
Test 5  PASS  Transposition consistency
```

---

## Open Issues

### Issue 1 — Policy cannot convert (HIGH PRIORITY)
**Symptom:** Greedy policy draws every game against a random opponent in 300
half-moves. Even MCTS(material) with two queens + rook took 73 moves to
deliver checkmate, spending many moves shuffling pieces while the opponent
king oscillated.

**Root cause:** Training data is Lichess games that end in **resignation**, not
checkmate. The model has almost never seen endgame conversion sequences —
"drive king to edge, deliver mate" — in its training data. The policy learned
to play natural-looking middlegame moves but was never rewarded for finishing
the job.

**Why it matters:** Self-play requires games to terminate with decisive outcomes.
Games drifting to move limit produce weak, noisy training signal. If the policy
can't convert, self-play will be dominated by draws and learn little.

**Hypotheses for step 2 draws (need to verify termination reasons):**
- Move limit (300 half-moves) hit before conversion
- 50-move rule triggering after pieces traded off
- Insufficient material draws (random play trades pieces, no mating material)

**Action:** Add termination reason breakdown to `run_match`. Verify which draw
type dominates. Consider endgame position injection in training data.

---

### Issue 2 — Forced mate convergence WARN (MEDIUM)
**Symptom:** Geometry probe (and value head) do not monotonically increase as
KQ vs K approaches mate. Value head outputs: +0.845, +0.865, +0.859, **+0.677**,
+0.873. The "Near mate" position (`8/8/8/8/7k/6Q1/8/6K1 w - - 0 1`) evaluates
lower than earlier positions consistently.

**Why it matters:** If the value head doesn't recognise proximity to mate, MCTS
won't prioritise mating lines. This directly contributes to step 2 draws.

**Hypotheses:**
- SF depth 15 labels for near-mate positions are imprecise (SF doesn't always
  see all the way to mate at depth 15, giving moderate cp scores instead of +∞)
- The specific "Near mate" FEN may be a poor test position — the queen on g2
  with king on h4 may not be as winning as it looks
- Model undertrained on endgame positions (Lichess games end by resignation,
  few training positions near forced mate)

**Action:** Verify the "Near mate" FEN with Stockfish directly. Consider adding
more endgame positions to the test suite, or weighting endgame positions in
future training.

---

### Issue 3 — Geometry probe signal weak (LOW / TRACK)
**Symptom:** R²=0.18 on game outcomes. Probe outputs in 0.04–0.07 range after
tanh squashing — too weak to guide MCTS effectively.

**Why it matters:** Step 6 ELO (-478) is partly caused by this. If geometry
carried stronger signal, the probe-based value function would be more useful.

**Note:** R²=0.18 is real and stable (train/val gap ~0.002). The geometry is not
a passenger — it encodes game-outcome information. But the signal strength needs
to grow through self-play iterations.

**Action:** Track R² across self-play rounds. Expect it to increase as self-play
reinforces winning/losing structure in the geometry.

---

### Issue 4 — Knight > Bishop in geometry (LOW)
**Symptom:** Test 1 fails strict monotonicity because Knight projection (0.0336)
> Bishop projection (0.0287). Test 2 passes all pairwise checks.

**Why it matters:** Borderline — N ≈ B in real chess. Not a fundamental problem.

**Action:** Monitor across iterations. Not worth fixing now.

---

## Philosophy

> "As long as the geometry gets better at each step, even if it is small, it
> makes sense to continue. Even if it stays the same. But we need to rid it of
> all obvious problems before pushing further." — Apr 10 2026

We are not comparing against the finished AlphaZero. We only see AlphaZero's
final integrated system — we don't know how its policy, value, or geometry
performed in isolation at this stage. Our ablation ladder gives diagnostic
visibility that AlphaZero's published results do not.

The gate (step 5) is passed. The foundation is real. Fix the obvious issues,
then self-play.

---

## Next Actions

### Before round 1 (do in order)

1. [ ] `bsub < jobs/gen_endgame.sh` — generate KQ vs K + KR vs K anchor dataset (~5 min)
2. [ ] `bsub < jobs/probe_geometry.sh` — baseline probe on `models/best.pt` before any self-play
       Record: effective rank, centroid cosine, separation gap. This is the comparison baseline.

### Round 1

3. [ ] `bsub < jobs/zigzag_r1.sh` — self-play → SF reeval (depth 12) → train → gate (200 games)
       Key settings: 16 workers, 500 games, n_sim=40, endgame anchor 20%, 3 seeds → best geometry
4. [ ] `bsub < jobs/probe_geometry_r1.sh` — post-round probe on `models/zigzag/r1/best.pt`

### Decision gate after round 1

- Separation gap HIGHER than baseline → geometry is developing, proceed to round 2
- Separation gap FLAT or WORSE → passenger problem active; increase anchor-frac to 0.30,
  consider expanding endgame stages (add stages 3 4 in gen_endgame.sh), or investigate
  explicit geometry loss (Fisher discriminant, LATER.md)

### After round 1 (diagnostic)

5. [ ] Re-run step 2 (Greedy vs Random) — expect improvement as policy sees checkmate positions
6. [ ] Re-run step 5 gate — track ELO delta vs material over rounds
7. [ ] Re-run test_geometry.py — track probe R² improvement

## Phase 1 history lesson (do not repeat)

The original Petra (Phase 1, R1–R4) ran 4 zigzag rounds and geometry stagnated.
ELO went 54.5% → 67% but only because n_sim increased (40 → 400) — not geometry.
R6 Tanh bottleneck cold start made geometry WORSE (centroid cosine 0.869 → 0.981).
Root cause: passenger problem — value head compensates without needing geometric structure.
Fix applied in doover: thin value head (no ReLU) + L2Norm bottleneck + endgame anchor.

## Closed

- [x] Step 6 debugging complete — geometry probe actively harmful at this stage,
      needs self-play to develop directional structure. Not a code bug.
