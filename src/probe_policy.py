"""
Policy head probe for PetraNet.

Diagnoses the state of the policy head — primarily by Top-k accuracy
against the SF best move stored in the dataset (move_idxs).

PRIMARY GATE: Top1 / Top5 vs SF best
  These measure whether masked argmax matches what MCTS would want
  to expand first. Random baseline at median n_legal=31 is ~1/31 ≈ 0.03.

    Top1 < 0.05  DEAD     (head not informative)
    Top1 < 0.20  WEAK     (above random but below MCTS-useful threshold)
    Top1 < 0.40  ALIVE    (MCTS-useful range)
    Top1 ≥ 0.40  STRONG

DIAGNOSTICS (not a gate):
  illegal_mass = 1 - sum(softmax(raw_logits)[legal_indices])
    With masked-CE training, illegal logits receive ZERO gradient
    (mask sets them to -inf in the loss; the targets are zero on
    illegal indices). illegal_mass therefore stays at the random
    baseline (~n_illegal / 4096 ≈ 0.995) regardless of how well
    the head learned. Useful only as a sanity check for unmasked
    training pipelines.

  masked_entropy / log(n_legal)
    How peaked the masked distribution is. Affects MCTS exploration
    breadth but a flat distribution can still match SF top-1 by a
    small margin — Top1 is the load-bearing metric, not entropy.

  per-phase breakdown
    opening (>26 pieces) / middle (12–26) / endgame (<12)

Usage
-----
    python3 src/probe_policy.py --model models/best.pt --dataset dataset.pt
    python3 src/probe_policy.py --model models/best.pt --dataset dataset.pt \
        --n 1000 --out results/probe_policy.json
"""

import argparse
import json
import os
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PetraNet
from board import board_to_tensor, move_to_index
from config import device


# ---------------------------------------------------------------------------
# Phase tagging
# ---------------------------------------------------------------------------

def phase_of(board: chess.Board) -> str:
    """Coarse phase tag from total piece count (incl. pawns and kings)."""
    n = sum(1 for _ in board.piece_map())
    if n > 26:
        return "opening"
    if n >= 12:
        return "middle"
    return "endgame"


# ---------------------------------------------------------------------------
# Per-position measurement
# ---------------------------------------------------------------------------

def probe_one(model: PetraNet, board: chess.Board, sf_best_idx: int) -> dict:
    """
    Run the policy head on a single board and extract:
      - top1_match / top5_match: argmax (top-5) of masked policy matches SF best
                                 None when sf_best_idx < 0 (SF eval failed)
      - illegal_mass: raw softmax probability on illegal-move indices (diagnostic)
      - masked_entropy: entropy of softmax over legal moves only (nats, diagnostic)
      - n_legal: number of legal moves
      - phase: coarse opening/middle/endgame tag
    """
    t = board_to_tensor(board).unsqueeze(0).to(device)
    _, logits = model.forward(t)
    logits = logits.squeeze(0)                              # (4096,)

    flip = (board.turn == chess.BLACK)
    legal_idx = torch.tensor(
        [move_to_index(m, flip=flip) for m in board.legal_moves],
        dtype=torch.long, device=device,
    )

    # Raw distribution over all 4096 outputs (diagnostic only)
    raw_probs = torch.softmax(logits, dim=0)
    legal_mass = raw_probs[legal_idx].sum().item()
    illegal_mass = 1.0 - legal_mass

    # Masked distribution — what MCTS actually sees
    mask = torch.full((4096,), float("-inf"), device=device)
    mask[legal_idx] = logits[legal_idx]
    masked_probs = torch.softmax(mask, dim=0)
    p = masked_probs[legal_idx]
    p = p[p > 0]
    masked_entropy = float(-(p * torch.log(p + 1e-12)).sum().item())

    # Top-k vs SF best — primary gate
    if sf_best_idx >= 0:
        top1_idx = int(masked_probs.argmax().item())
        top1_match = (top1_idx == sf_best_idx)
        k = min(5, int(legal_idx.numel()))
        top5_idxs = masked_probs.topk(k).indices.tolist()
        top5_match = sf_best_idx in top5_idxs
    else:
        top1_match = None
        top5_match = None

    return {
        "top1_match":     top1_match,
        "top5_match":     top5_match,
        "illegal_mass":   float(illegal_mass),
        "masked_entropy": masked_entropy,
        "n_legal":        int(legal_idx.numel()),
        "max_entropy":    float(np.log(legal_idx.numel())),
        "phase":          phase_of(board),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def percentiles(xs):
    a = np.asarray(xs)
    return {
        "p10":  float(np.percentile(a, 10)),
        "p50":  float(np.percentile(a, 50)),
        "p90":  float(np.percentile(a, 90)),
        "mean": float(a.mean()),
    }


def _hit_rate(rows, key):
    """Mean of boolean key, skipping rows where the value is None."""
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def summarise(rows: list) -> dict:
    """Produce overall and per-phase summaries."""
    illegal = [r["illegal_mass"]   for r in rows]
    entropy = [r["masked_entropy"] for r in rows]
    # Normalised entropy: actual / max-possible — comparable across n_legal.
    norm_entropy = [
        r["masked_entropy"] / r["max_entropy"] if r["max_entropy"] > 0 else 0.0
        for r in rows
    ]
    n_with_sf = sum(1 for r in rows if r["top1_match"] is not None)

    overall = {
        "n":                      len(rows),
        "n_with_sf_target":       n_with_sf,
        "top1_hit":               _hit_rate(rows, "top1_match"),
        "top5_hit":               _hit_rate(rows, "top5_match"),
        "illegal_mass":           percentiles(illegal),
        "masked_entropy":         percentiles(entropy),
        "masked_entropy_norm":    percentiles(norm_entropy),
        "median_n_legal":         float(np.median([r["n_legal"] for r in rows])),
    }

    by_phase = {}
    for ph in ("opening", "middle", "endgame"):
        sub = [r for r in rows if r["phase"] == ph]
        if not sub:
            continue
        by_phase[ph] = {
            "n":                   len(sub),
            "top1_hit":            _hit_rate(sub, "top1_match"),
            "top5_hit":            _hit_rate(sub, "top5_match"),
            "illegal_mass":        percentiles([r["illegal_mass"]   for r in sub]),
            "masked_entropy":      percentiles([r["masked_entropy"] for r in sub]),
            "masked_entropy_norm": percentiles(
                [r["masked_entropy"] / r["max_entropy"] if r["max_entropy"] > 0 else 0.0
                 for r in sub]
            ),
        }

    return {"overall": overall, "by_phase": by_phase}


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def verdict(summary: dict) -> str:
    """Translate the headline number into a one-line diagnosis."""
    top1 = summary["overall"]["top1_hit"]
    top5 = summary["overall"]["top5_hit"]

    if top1 is None:
        return "NO SF TARGETS (cannot grade — dataset move_idxs all -1)"

    if top1 < 0.05:
        band = "DEAD     (Top1 ≈ random — head not informative)"
    elif top1 < 0.20:
        band = "WEAK     (above random but below MCTS-useful threshold)"
    elif top1 < 0.40:
        band = "ALIVE    (MCTS-useful range)"
    else:
        band = "STRONG   (Top1 ≥ 0.40)"

    return f"{band}  |  Top1={top1:.3f}  Top5={top5:.3f}"


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _fmt_pct(d):
    return (f"p10={d['p10']:.3f}  p50={d['p50']:.3f}  "
            f"p90={d['p90']:.3f}  mean={d['mean']:.3f}")


def _fmt_hit(x):
    return f"{x:.3f}" if x is not None else "n/a"


def print_summary(summary: dict, verdict_str: str):
    o = summary["overall"]
    print("\n" + "=" * 68)
    print("POLICY PROBE — overall")
    print("=" * 68)
    print(f"  n positions       : {o['n']}  (with SF target: {o['n_with_sf_target']})")
    print(f"  median n_legal    : {o['median_n_legal']:.0f}")
    print(f"  Top1 vs SF best   : {_fmt_hit(o['top1_hit'])}   (random ≈ 1/median_n_legal)")
    print(f"  Top5 vs SF best   : {_fmt_hit(o['top5_hit'])}")
    print(f"  -- diagnostics (not gates) --")
    print(f"  illegal_mass      : {_fmt_pct(o['illegal_mass'])}")
    print(f"  masked entropy    : {_fmt_pct(o['masked_entropy'])}")
    print(f"  entropy / log(n_legal) : {_fmt_pct(o['masked_entropy_norm'])}")

    print("\n  by phase:")
    for ph, s in summary["by_phase"].items():
        print(f"    {ph:8s}  n={s['n']:4d}  "
              f"Top1={_fmt_hit(s['top1_hit'])}  "
              f"Top5={_fmt_hit(s['top5_hit'])}  "
              f"illegal_p50={s['illegal_mass']['p50']:.3f}  "
              f"norm_ent_p50={s['masked_entropy_norm']['p50']:.3f}")

    print("\n  VERDICT: " + verdict_str)
    print("=" * 68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True, help="Path to model .pt")
    ap.add_argument("--dataset", required=True, help="Path to dataset .pt")
    ap.add_argument("--n",       type=int, default=1000,
                    help="Number of val positions to probe (default 1000)")
    ap.add_argument("--out",     default=None,
                    help="Optional JSON output path")
    args = ap.parse_args()

    model = PetraNet().to(device)
    sd = torch.load(args.model, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  New/untrained parameters in model definition: {missing}")
    if unexpected:
        print(f"  Unexpected checkpoint parameters ignored: {unexpected}")
    model.eval()
    print(f"Loaded model from {args.model}")

    data  = torch.load(args.dataset, map_location="cpu", weights_only=False)
    split = data["val"]
    fens  = split["fens"][:args.n]
    move_idxs = split["move_idxs"][:args.n].tolist() if "move_idxs" in split else [-1] * len(fens)
    print(f"Probing {len(fens)} val positions from {args.dataset}")

    rows = []
    for i, fen in enumerate(fens):
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if board.is_game_over() or not any(board.legal_moves):
            continue
        rows.append(probe_one(model, board, sf_best_idx=int(move_idxs[i])))
        if (i + 1) % 100 == 0:
            print(f"  [{i+1:5d}/{len(fens)}]")

    summary = summarise(rows)
    v = verdict(summary)
    print_summary(summary, v)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "model":    args.model,
                "dataset":  args.dataset,
                "n":        len(rows),
                "summary":  summary,
                "verdict":  v,
            }, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
