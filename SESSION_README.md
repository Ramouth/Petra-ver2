# Petra Doover — Session Package
Generated April 2026.

## What's in here

### src/ — Complete source (doover branch)
All files from the doover branch including:
- board.py          — STM board flip + flip-aware move encoding (ROOT FIX)
- model.py          — L2 bottleneck, thin value head, flip-aware policy
- data.py           — Flip-aware move encoding
- selfplay.py       — Flip fixes, resign ordering, MAX_POSITIONS=50
- train.py          — Memory fix in mix_anchor, scheduler patience
- test_geometry.py  — Test 3 rewritten (STM symmetry), balanced anchors
- probe_geometry.py — STM values, effective rank, label bucketing
- evaluate.py       — Balanced anchors, gate exit code wired
- zigzag.py         — Geometry probe in loop, dead code removed
- select_geometry.py — A/B geometry selection by separation gap
- concepts.py       — Concept anchoring stub (8 chess concepts)
- mcts.py           — Unchanged (correct as-is)
- config.py         — Unchanged
- reeval_stockfish.py — Unchanged
- gen_openings.py   — Unchanged
- compare_geometry.py — Unchanged

### Documentation
- FIX.md      — Full code review findings + doover rationale
- NOW.md      — Topology monitor + alignment/uniformity (before next HPC run)
- LATER.md    — vMF loss, Fréchet mean, Fisher discriminant (after clean geometry)
- ZIGZAG.md   — Zigzag training curriculum
- ROADMAP.md  — Project roadmap

## To apply on your machine

```bash
git checkout -b doover
cp petra_doover/src/*.py src/
cp petra_doover/*.md .
git add src/ *.md
git commit -m "Doover: board flip, L2 geometry, full code review"
git push -u origin doover
```

## First HPC run

```bash
# Train 3 seeds in parallel
python3 src/train.py --dataset data/sf_r1.pt --out models/cand_s0 --seed 0 &
python3 src/train.py --dataset data/sf_r1.pt --out models/cand_s1 --seed 1 &
python3 src/train.py --dataset data/sf_r1.pt --out models/cand_s2 --seed 2 &
wait

# Select best geometry
python3 src/select_geometry.py \
    --candidates models/cand_s0/best.pt models/cand_s1/best.pt models/cand_s2/best.pt \
    --dataset data/sf_r1.pt \
    --out models/zigzag/r1/best.pt

# Gate evaluation on winner
python3 src/evaluate.py --model models/zigzag/r1/best.pt --step 5 --games 100
```

## Install ripser before the run

```bash
pip install ripser persim --break-system-packages
```

## The key numbers to watch

After training, run test_geometry.py:
- Test 3 (STM symmetry): should be 5/5 (was 3/5)
- Separation gap (Check 2): target > 0.10 (was ~0.048)
- Effective rank (Check 1): target > 30 (was 26/128 dead dims)
- Black-up-queen sanity: should pass (was failing)

If 5/5 → geometry is working → proceed to GPU.
