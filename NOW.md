# NOW.md — Phase 2: Sharpening Geometry, Building Policy & ELO

*Phase 1 closed 2026-05-01.*

**Phase 1 result:** Project forked into petra-elo (champion: `natural`) and petra-drawness (champion: `dr03`). dr03 is the first model with all three centroid pairs anti-aligned + drawness PoC pass + healthy rank. Both score identically at SF skill=0 (~980 ELO absolute, no opening book, undertrained policy).

**Phase 2 goal:** turn the geometry breakthrough into a working engine — sharpen the value geometry, give policy a real treatment, convert that into measurable ELO. Three independent workstreams.

---

## 1. Sharpen geometry — value-only training

Test whether dr03's three-pole structure means training fights itself less.

- Three arms (init from `dr03` / `natural` / cold), identical hyperparams, same fresh SF-reeval data slice.
- **Value-only** training (`policy_weight=0`). See `memory/feedback_separate_policy_value.md`.
- Per-epoch instrumentation: value loss + variance, cos(W,L) / cos(W,D) / cos(D,L), effective rank, draw-probe AUC on PoC validator FENs.
- Decision: dr03 holds three poles + lower/smoother loss → launchpad. dr03 collapses to two poles → fragile local optimum.

Cost: ~1 day wall, single bsub per arm.

## 2. Build policy — frozen trunk + new head

Prerequisite: policy is dead — no legal-move masking + human-move targets after SF reeval. See `memory/project_policy_dead.md`. Fix both.

- Freeze dr03's trunk. Train a fresh policy head on top.
- Legal-move masking in the head + inference path.
- MCTS-style soft targets, not raw human moves.
- Same SF-reeval data slice as §1.
- Sanity gate: bounded policy entropy, only legal moves at inference.

Fallback if frozen-trunk policy plateaus: two fully separate networks (value + policy), same data, ~2x inference cost. Decide after first round.

## 3. Build ELO — opening book + full bracket calibration

Current: dr03 ≈ natural ≈ ~980 ELO at SF skill=0 (n_sim=100, no book). See `memory/project_elo_calibration_skill0.md`.

Near-term:
- Add Polyglot-style opening book to inference.
- Re-run `benchmark_elo_stockfish.py` full bracket (skill=0/5/10).
- Bump bsub `-W 16:00` or split skill levels into separate jobs (8h was insufficient — ~3 min/game × 300 games ≈ 15h wall).

Medium-term (after §2 lands):
- Self-play smoke test: dr03 + new policy plays itself, n_sim=400, ~1000 games. Eyeball W/D/L distribution and game length.
- One AlphaZero-style iteration: train on self-play + SF-supervised mix, ELO benchmark vs current best.
- Side benefit: balanced self-play between two Petras should produce more drawn games than the elite human corpus does → feeds the drawness fork directly.

---

## Success criteria

- §1: clear yes/no on whether dr03's three poles survive value-only training and converge with less fighting than natural.
- §2: working policy head (legal moves only, bounded entropy) that doesn't perturb dr03's value geometry.
- §3: defensible absolute-ELO number for the paper (full bracket completed, with opening book).

Phase 3 begins if §3 self-play closes a positive ΔELO loop.

---

*Detail lives in `jobs/`, `memory/`, and per-experiment scripts. This file is a navigation marker, not a spec.*
