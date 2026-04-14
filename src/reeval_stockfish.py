"""
Re-evaluate dataset positions with Stockfish.

Loads an existing dataset.pt, evaluates each position with Stockfish at a
fixed depth, replaces game-outcome labels with engine evaluations, and saves
a new dataset_sf.pt ready for training.

Label convention (matches training pipeline):
  Stockfish centipawn eval is converted to the side-to-move perspective
  and squashed through tanh(cp / 400) → range (-1, +1).
  +1 = side to move is winning decisively
   0 = equal
  -1 = side to move is losing decisively

  Mate scores: mapped to ±1.0 directly.

Why tanh(cp / 400)?
  400cp is roughly one piece advantage. tanh saturates gracefully so a
  +800cp position doesn't dominate the MSE loss over a +200cp position.
  AlphaZero uses a similar squashing convention.

Usage
-----
    python3 src/reeval_stockfish.py --dataset dataset.pt --out dataset_sf.pt
    python3 src/reeval_stockfish.py --dataset dataset.pt --out dataset_sf.pt \\
        --depth 12 --n 200000 --stockfish /usr/games/stockfish --workers 32
"""

import argparse
import math
import multiprocessing
import os
import subprocess
import sys
import time

import chess
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board import move_to_index


# ---------------------------------------------------------------------------
# Stockfish interface (minimal, no chess library dependency)
# ---------------------------------------------------------------------------

class Stockfish:
    """
    Thin wrapper around a Stockfish subprocess using UCI protocol.
    Thread-unsafe — one instance per process.
    """

    def __init__(self, path: str = "/usr/games/stockfish", depth: int = 10, threads: int = 1):
        self.depth = depth
        self._proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send("isready")
        self._wait_for("readyok")
        self._send(f"setoption name Threads value {threads}")

    def _send(self, cmd: str):
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _wait_for(self, token: str) -> list:
        lines = []
        while True:
            line = self._proc.stdout.readline()
            if not line:  # EOF — process died
                stderr = self._proc.stderr.read()
                raise RuntimeError(
                    f"Stockfish process exited before sending {token!r}.\n"
                    f"stderr: {stderr.strip() or '(empty)'}"
                )
            line = line.strip()
            lines.append(line)
            if token in line:
                return lines

    def evaluate(self, fen: str) -> tuple:
        """
        Evaluate a position given its FEN string.
        Returns (value, bestmove_uci) where:
          value       : float in (-1, +1) from the side-to-move's perspective
          bestmove_uci: UCI move string (e.g. 'e2e4'), or '(none)' if no legal moves
        """
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        lines = self._wait_for("bestmove")

        cp = None
        mate = None
        bestmove = "(none)"
        for line in lines:
            if line.startswith("info") and "score" in line:
                parts = line.split()
                try:
                    score_idx = parts.index("score")
                    score_type = parts[score_idx + 1]
                    score_val  = int(parts[score_idx + 2])
                    if score_type == "cp":
                        cp = score_val
                    elif score_type == "mate":
                        mate = score_val
                except (ValueError, IndexError):
                    pass
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) > 1:
                    bestmove = parts[1]

        if mate is not None:
            return (1.0 if mate > 0 else -1.0), bestmove
        if cp is not None:
            return math.tanh(cp / 400.0), bestmove
        return 0.0, bestmove

    def close(self):
        self._send("quit")
        self._proc.wait()


# ---------------------------------------------------------------------------
# Multiprocessing worker (one Stockfish instance per worker process)
# ---------------------------------------------------------------------------

_worker_sf = None

def _init_worker(sf_path: str, depth: int):
    global _worker_sf
    _worker_sf = Stockfish(path=sf_path, depth=depth, threads=1)

def _eval_one(args):
    idx, fen = args
    try:
        val, bestmove_uci = _worker_sf.evaluate(fen)
        board = chess.Board(fen)
        flip = (board.turn == chess.BLACK)

        # SF best move index in STM-relative coordinates
        sf_move_idx = -1
        if bestmove_uci and bestmove_uci != "(none)":
            try:
                move = chess.Move.from_uci(bestmove_uci)
                sf_move_idx = move_to_index(move, flip=flip)
            except Exception:
                pass

        # Legal move mask bit-packed: 4096 bits → 512 bytes per position
        mask = torch.zeros(4096, dtype=torch.uint8)
        for m in board.legal_moves:
            mask[move_to_index(m, flip=flip)] = 1
        packed_mask = torch.packbits(mask)   # shape (512,)

        return idx, val, sf_move_idx, packed_mask, False
    except Exception:
        return idx, 0.0, -1, None, True


# ---------------------------------------------------------------------------
# Re-evaluation
# ---------------------------------------------------------------------------

def reeval(dataset_path: str,
           out_path: str,
           stockfish_path: str = "/usr/games/stockfish",
           depth: int = 10,
           n: int = None,
           seed: int = 42,
           workers: int = 1,
           min_decisive: float = 0.0,
           max_pieces: int = 32):
    """
    Load dataset, re-evaluate positions with Stockfish, save new dataset.

    n:             if set, subsample this many positions from train+val combined.
    workers:       number of parallel Stockfish processes.
    min_decisive:  keep only positions where |tanh(eval)| >= this value.
                   0.0 = keep all. 0.3 ≈ 130cp advantage.
    max_pieces:    keep only positions with at most this many pieces on the board.
                   32 = keep all. 16 = endgame-like. 24 = simplified middlegame.
                   Filters out tactically complex positions a baby model can't learn from.
    """
    print(f"Loading {dataset_path} ...")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)

    # Merge train + val for parallel evaluation, but track the boundary so
    # the original split can be restored. This preserves game-level no-leakage:
    # positions from the same game stay in the same split, so val loss is not
    # optimistic due to positions from the same game appearing in both splits.
    train_d = data["train"]
    val_d   = data["val"]
    n_train_orig = len(train_d["tensors"])

    all_tensors  = torch.cat([train_d["tensors"],  val_d["tensors"]],  dim=0)
    all_fens     = train_d["fens"] + val_d["fens"]
    all_moves    = torch.cat([train_d["move_idxs"], val_d["move_idxs"]], dim=0)

    # Carry visit_dists through if present — these are the MCTS policy targets
    # from self-play and must survive reeval so train.py uses the dense loss.
    # Supervised datasets won't have them; that's fine (train.py handles both).
    has_vd = "visit_dists" in train_d
    if has_vd:
        all_visit_dists = torch.cat([train_d["visit_dists"], val_d["visit_dists"]], dim=0)
    else:
        all_visit_dists = None

    # Free the full loaded dataset before starting the Stockfish loop.
    del data, train_d, val_d
    import gc; gc.collect()

    total = len(all_fens)
    print(f"  Total positions: {total:,}")

    # Subsample if requested — sample within each split to preserve boundaries.
    if n is not None and n < total:
        import random
        rng = random.Random(seed)
        sampled_idxs = sorted(rng.sample(range(total), n))
        all_tensors = all_tensors[sampled_idxs]
        all_fens    = [all_fens[i] for i in sampled_idxs]
        all_moves   = all_moves[sampled_idxs]
        if all_visit_dists is not None:
            all_visit_dists = all_visit_dists[sampled_idxs]
        print(f"  Subsampled to {n:,} positions")
    else:
        sampled_idxs = list(range(total))
        n = total

    print(f"\nStarting {workers} Stockfish worker(s) (depth={depth}) ...")

    new_values    = torch.zeros(n, dtype=torch.float32)
    new_move_idxs = all_moves.clone()           # fallback: original human moves
    all_packed_masks = torch.zeros(n, 512, dtype=torch.uint8)
    valid_mask    = torch.zeros(n, dtype=torch.bool)
    t0 = time.time()
    errors = 0
    sf_move_hits = 0
    done = 0

    with multiprocessing.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(stockfish_path, depth),
    ) as pool:
        print(f"  Workers ready.")
        for idx, val, sf_move_idx, packed_mask, err in pool.imap(
                _eval_one, enumerate(all_fens), chunksize=64):
            new_values[idx] = val
            if not err:
                if sf_move_idx >= 0:
                    new_move_idxs[idx] = sf_move_idx
                    sf_move_hits += 1
                if packed_mask is not None:
                    all_packed_masks[idx] = packed_mask
                    valid_mask[idx] = True
            else:
                errors += 1
            done += 1
            if done % 5000 == 0:
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (n - done) / rate
                print(f"  {done:>8,} / {n:,}  ({rate:.0f} pos/s)  ETA {eta/60:.0f} min")

    elapsed = time.time() - t0
    print(f"\nDone: {n:,} positions in {elapsed:.0f}s  ({n/elapsed:.0f} pos/s)")
    if errors:
        print(f"WARNING: {errors} evaluation errors (set to 0.0)")
    print(f"SF best moves used: {sf_move_hits:,} / {n:,}  "
          f"(fallback to human moves: {n - sf_move_hits:,})")
    has_masks = valid_mask.all().item()
    if not has_masks:
        n_missing = int((~valid_mask).sum().item())
        print(f"WARNING: {n_missing} positions missing legal masks — masks will not be stored")

    # Report label distribution
    import numpy as np
    vals = new_values.numpy()
    print(f"\nLabel statistics (SF evals, tanh-squashed):")
    print(f"  Mean:   {vals.mean():.4f}")
    print(f"  Std:    {vals.std():.4f}")
    print(f"  Min:    {vals.min():.4f}")
    print(f"  Max:    {vals.max():.4f}")
    print(f"  |v|>0.5 (decisive): {(np.abs(vals) > 0.5).mean()*100:.1f}%")
    print(f"  |v|<0.1 (equal):    {(np.abs(vals) < 0.1).mean()*100:.1f}%")

    # ---------------------------------------------------------------------------
    # Filter: decisive + structurally simple
    #
    # Two criteria (both must pass):
    #   1. |eval| >= min_decisive  — the position is actually winning/losing
    #   2. piece_count <= max_pieces — the advantage is visible on the board,
    #      not hidden in deep tactics. Fewer pieces = endgame-like = learnable.
    #
    # A forced mate-in-15 through a tactical combination scores 1.0 but looks
    # like a normal middlegame to a baby model — it's noise, not signal.
    # A queen-up position with 8 pieces left is immediately learnable.
    # ---------------------------------------------------------------------------
    if min_decisive > 0.0 or max_pieces < 32:
        import chess
        keep = []
        skipped_eval = skipped_pieces = 0
        for i, fen in enumerate(all_fens):
            v = float(new_values[i])
            if abs(v) < min_decisive:
                skipped_eval += 1
                continue
            board = chess.Board(fen)
            pc = bin(int(board.occupied)).count("1")
            if pc > max_pieces:
                skipped_pieces += 1
                continue
            keep.append(i)

        print(f"\nFilter applied:")
        print(f"  min_decisive={min_decisive}  max_pieces={max_pieces}")
        print(f"  Kept:            {len(keep):,} / {n:,}")
        print(f"  Dropped (eval):  {skipped_eval:,}")
        print(f"  Dropped (pieces):{skipped_pieces:,}")

        if len(keep) < 100:
            raise ValueError(
                f"Filter too aggressive — only {len(keep)} positions remain. "
                f"Lower min_decisive or raise max_pieces."
            )

        keep_t           = torch.tensor(keep, dtype=torch.long)
        new_values       = new_values[keep_t]
        all_tensors      = all_tensors[keep_t]
        all_moves        = all_moves[keep_t]
        new_move_idxs    = new_move_idxs[keep_t]
        all_packed_masks = all_packed_masks[keep_t]
        valid_mask       = valid_mask[keep_t]
        all_fens         = [all_fens[i] for i in keep]
        sampled_idxs     = [sampled_idxs[i] for i in keep]
        if all_visit_dists is not None:
            all_visit_dists = all_visit_dists[keep_t]
        has_masks = valid_mask.all().item()

        filtered_vals = new_values.numpy()
        print(f"  Post-filter decisive rate: "
              f"{(np.abs(filtered_vals) > 0.5).mean()*100:.1f}%")

    # Restore original train/val split by checking each position's source index.
    # Positions with sampled_idxs[i] < n_train_orig came from the original train
    # split; the rest came from val. This preserves game-level no-leakage.
    train_idxs = [i for i, orig in enumerate(sampled_idxs) if orig < n_train_orig]
    val_idxs   = [i for i, orig in enumerate(sampled_idxs) if orig >= n_train_orig]
    if not val_idxs:
        # Edge case: subsampling drew only train positions — take last 5% as val.
        n_val      = max(1, len(train_idxs) // 20)
        val_idxs   = train_idxs[-n_val:]
        train_idxs = train_idxs[:-n_val]

    def pack(subset_idxs):
        d = {
            "tensors":   all_tensors[subset_idxs],
            "values":    new_values[subset_idxs],
            "move_idxs": new_move_idxs[subset_idxs],   # SF best moves
            "fens":      [all_fens[i] for i in subset_idxs],
        }
        if all_visit_dists is not None:
            d["visit_dists"] = all_visit_dists[subset_idxs]
        if has_masks:
            d["legal_masks"] = all_packed_masks[subset_idxs]   # (N, 512) uint8, bit-packed
        return d

    out = {
        "train": pack(train_idxs),
        "val":   pack(val_idxs),
        "meta": {
            "source":          dataset_path,
            "stockfish_depth": depth,
            "n_train":         len(train_idxs),
            "n_val":           len(val_idxs),
            "label_type":      "stockfish_tanh_cp400",
            "has_visit_dists": has_vd,
        },
    }

    torch.save(out, out_path)
    print(f"\nSaved → {out_path}")
    print(f"  train: {len(train_idxs):,}  val: {len(val_idxs):,}")
    if has_vd:
        print(f"  visit_dists: preserved (dense policy loss active in train.py)")
    else:
        print(f"  visit_dists: absent (supervised dataset — one-hot policy loss)")
    print(f"  move_idxs:   SF best moves  (human-move fallback where SF unavailable)")
    if has_masks:
        print(f"  legal_masks: stored ({512} bytes/position bit-packed, "
              f"train.py will mask policy logits)")
    else:
        print(f"  legal_masks: NOT stored (some positions had errors)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    required=True,                   help="Input dataset.pt")
    ap.add_argument("--out",        default="dataset_sf.pt",         help="Output path")
    ap.add_argument("--stockfish",  default="/usr/games/stockfish",  help="Stockfish binary")
    ap.add_argument("--depth",      type=int, default=10,            help="Search depth")
    ap.add_argument("--n",          type=int, default=200_000,
                    help="Number of positions to evaluate (default 200k)")
    ap.add_argument("--seed",          type=int,   default=42)
    ap.add_argument("--workers",       type=int,   default=1,
                    help="Number of parallel Stockfish processes (default 1)")
    ap.add_argument("--min-decisive",  type=float, default=0.0,
                    help="Keep only positions with |tanh(eval)| >= this. "
                         "0.3 ≈ 130cp. 0.0 = keep all (default).")
    ap.add_argument("--max-pieces",    type=int,   default=32,
                    help="Keep only positions with <= this many pieces. "
                         "32 = all, 24 = simplified middlegame, 16 = endgame (default: 32).")
    args = ap.parse_args()

    reeval(
        dataset_path=args.dataset,
        out_path=args.out,
        stockfish_path=args.stockfish,
        depth=args.depth,
        n=args.n,
        seed=args.seed,
        workers=args.workers,
        min_decisive=args.min_decisive,
        max_pieces=args.max_pieces,
    )


if __name__ == "__main__":
    main()
