"""
Policy head probe for PetraNet.

Diagnoses the state of the policy head BEFORE building the new policy
training pipeline (Phase 2 §2). Inference is masked in model.policy()
but training is unmasked — so the raw 4096-dim logit distribution may
have most of its mass on illegal moves, with the masked output being
just whatever leaks through.

This probe answers three questions:

  1. Where is the raw logit mass?
       illegal_mass = 1 - sum(softmax(raw_logits)[legal_indices])
       0.95+   policy is essentially predicting illegal moves
       0.5–0.95 partial leakage
       <0.5    head has learned legality despite no training signal

  2. Is the masked policy alive or collapsed?
       entropy of masked softmax over legal moves only
       very low → collapsed (always argmax to one move regardless)
       very high → alive but undirected (uniform over legal)
       mid-range with variance across positions → contextual

  3. Does the picture vary by phase?
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

def probe_one(model: PetraNet, board: chess.Board) -> dict:
    """
    Run the policy head on a single board and extract:
      - illegal_mass: raw softmax probability on illegal-move indices
      - masked_entropy: entropy of softmax over legal moves only (nats)
      - n_legal: number of legal moves (denominator for entropy interpretation)
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

    # Raw distribution over all 4096 outputs
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

    return {
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


def summarise(rows: list) -> dict:
    """Produce overall and per-phase summaries."""
    illegal = [r["illegal_mass"]   for r in rows]
    entropy = [r["masked_entropy"] for r in rows]
    # Normalised entropy: actual / max-possible — comparable across n_legal.
    norm_entropy = [
        r["masked_entropy"] / r["max_entropy"] if r["max_entropy"] > 0 else 0.0
        for r in rows
    ]

    overall = {
        "n":                      len(rows),
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
    """Translate the headline numbers into a one-line diagnosis."""
    illegal_p50 = summary["overall"]["illegal_mass"]["p50"]
    norm_ent_p50 = summary["overall"]["masked_entropy_norm"]["p50"]

    parts = []

    if illegal_p50 > 0.95:
        parts.append("DEAD (raw logits dominated by illegal moves)")
    elif illegal_p50 > 0.5:
        parts.append("LEAKY (substantial illegal-move mass)")
    else:
        parts.append("LEGAL-AWARE (most raw mass on legal moves)")

    if norm_ent_p50 > 0.9:
        parts.append("UNDIRECTED (masked entropy near uniform)")
    elif norm_ent_p50 < 0.2:
        parts.append("COLLAPSED (masked entropy near zero)")
    else:
        parts.append("CONTEXTUAL (masked entropy mid-range)")

    return " + ".join(parts)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _fmt_pct(d):
    return (f"p10={d['p10']:.3f}  p50={d['p50']:.3f}  "
            f"p90={d['p90']:.3f}  mean={d['mean']:.3f}")


def print_summary(summary: dict, verdict_str: str):
    o = summary["overall"]
    print("\n" + "=" * 68)
    print("POLICY PROBE — overall")
    print("=" * 68)
    print(f"  n positions      : {o['n']}")
    print(f"  median n_legal   : {o['median_n_legal']:.0f}")
    print(f"  illegal_mass     : {_fmt_pct(o['illegal_mass'])}")
    print(f"  masked entropy   : {_fmt_pct(o['masked_entropy'])}")
    print(f"  entropy / log(n_legal) : {_fmt_pct(o['masked_entropy_norm'])}")

    print("\n  by phase:")
    for ph, s in summary["by_phase"].items():
        print(f"    {ph:8s}  n={s['n']:4d}  "
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
    print(f"Probing {len(fens)} val positions from {args.dataset}")

    rows = []
    for i, fen in enumerate(fens):
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if board.is_game_over() or not any(board.legal_moves):
            continue
        rows.append(probe_one(model, board))
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
