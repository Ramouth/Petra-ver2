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

## Next Steps
1. `bsub < jobs/probe_geometry.sh` — baseline geometry probe on `sf_gpu/best.pt`
2. `bsub < jobs/zigzag_r1.sh` — round 1: self-play (n_sim=40, SF depth 12, 500 games)
   - Opening book generated at runtime to avoid OOD opening bias
3. `bsub < jobs/probe_geometry_r1.sh` — compare geometry vs baseline
