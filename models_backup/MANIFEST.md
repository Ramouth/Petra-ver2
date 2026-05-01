# models_backup — MANIFEST

Local backup of Petra champion model checkpoints. See `../BACKUP.md` for
the procedure. Each file is `<model_name>.best.pt` — flattened from the
HPC layout `models/<model_name>/best.pt`.

| Filename | Track | Status | Origin (HPC path) | Recipe |
|---|---|---|---|---|
| `drawness_full_natural_init_dr03.best.pt` | petra-drawness | **CHAMPION** (2026-05-01) | `~/Petra-ver2/models/drawness_full_natural_init_dr03/best.pt` | BCE λ=1.0 + draw-reg **0.3** + drawness curriculum, init `2021_06_all`, lr 3e-4, 18 epochs |
| `drawness_full_natural_init.best.pt` | petra-drawness | superseded by dr03 | `~/Petra-ver2/models/drawness_full_natural_init/best.pt` | BCE λ=1.0 + draw-reg 1.0 + drawness curriculum, init `2021_06_all`, lr 3e-4, 20 epochs |
| `drawness_full_natural_init_dr05.best.pt` | petra-drawness | recipe ablation | `~/Petra-ver2/models/drawness_full_natural_init_dr05/best.pt` | Same as dr03 but draw-reg 0.5; collapsed win/loss; documents the bistable trade |
| `natural.best.pt` | petra-elo | **CHAMPION** | `~/Petra-ver2/models/natural/best.pt` | value+policy+rank-reg, drawness curriculum, NO drawness scaffold, init `2021_06_all`, lr 3e-4 |
| `2021_06_all.best.pt` | base | init for natural lineage | `~/Petra-ver2/models/2021_06_all/best.pt` | n/a — base supervised model on Lichess 2021-06 mixed-elo |
| `drawness_full.best.pt` | drawness predecessor | not champion | `~/Petra-ver2/models/drawness_full/best.pt` | Same recipe as champion but init `phase15_mid_no_endgame` (dead policy), policy-weight=0 |
| `feb_sf.best.pt` | elo predecessor | not champion | `~/Petra-ver2/models/feb_sf/best.pt` | SF-relabeled Feb 2020 |
| `natural_v2.best.pt` | elo alternative | tied with natural | `~/Petra-ver2/models/natural_v2/best.pt` | Same recipe family, larger corpus (1.99M vs 453k) |
| `natural_with_drawness.best.pt` | drawness Path A artifact | negative result | `~/Petra-ver2/models/natural_with_drawness/best.pt` | natural backbone (frozen) + post-hoc sklearn LR drawness head (50k synthetic pairs) |
| `drawness_head_v2.best.pt` | drawness predecessor | not champion | `~/Petra-ver2/models/drawness_head_v2/best.pt` | Frozen `phase15_mid_no_endgame` backbone + post-hoc fit drawness head |

## Champion summary (2026-05-01)

**petra-drawness — `drawness_full_natural_init_dr03.best.pt` (NEW CHAMPION)**
- PoC battery: centroid AUC 0.663, PC1 PASS 0.656, logreg 0.664, d=+0.59
- Drawness gates: 3/4 (KR 0.755 ✓ / KNN 0.453 ✗ / Sicilian 0.030 ✓ / KQ 0.054 ✓)
- **Three-pole geometry**: win·loss strict −0.34, win·draw strict −0.43, loss·draw strict −0.41 (all three negative — first in project)
- Effective rank: 28.8
- Top1: 0.230
- β1: 150
- ELO H2H vs natural: pending (expected 45-52% wr based on +17% Top1 over dr10)

**Predecessor — `drawness_full_natural_init.best.pt` (dr10)**
- PoC battery: centroid 0.664, PC1 0.666, logreg 0.665, d=+0.62 (tied with dr03)
- Drawness gates: 4/4 (slightly stronger head)
- Win·loss collapsed (+0.10), rank 23.1, Top1 0.197
- ELO H2H vs natural: 39.8% wr (−72 ELO)

**petra-elo — `natural.best.pt`**
- PoC battery: centroid 0.589, PC1 FAIL 0.493, logreg 0.607, d=+0.32
- Drawness gates: 0/4 (head untrained)
- Win·loss cosine (probe, strict): −0.65
- Effective rank: 38.7
- Top1: 0.247
- ELO H2H: +143 ELO vs `2021_06_all`, +121 wins for `natural_v2`

## Reproducibility

The training datasets used for the champions:

- `dataset_drawness_curriculum.pt`: 477k positions (453k train + 24k val)
  - Filter: outcome=draw AND |SF|<0.15 AND ply≥40 + matched decisive negatives
  - Source: Lichess 2021-06 ≥2300 elo
  - Backed up in priority 3 (see ../BACKUP.md)

The training data is reproducible from the Lichess monthly PGN archives
+ the `src/build_drawness_curriculum.py` script with the parameters above.

## Storage status

Mark each row with `[ ]` (not yet pulled) → `[X]` (local backup verified).

- [ ] drawness_full_natural_init.best.pt
- [ ] natural.best.pt
- [ ] 2021_06_all.best.pt
- [ ] drawness_full.best.pt
- [ ] feb_sf.best.pt
- [ ] natural_v2.best.pt
- [ ] natural_with_drawness.best.pt
- [ ] drawness_head_v2.best.pt

External backup location (fill in once chosen):
- [ ] GitHub Git LFS: <url>
- [ ] Cloud (specify): <link>
