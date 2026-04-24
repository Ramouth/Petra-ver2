"""
Generate material imbalance positions for the Phase 1.5 bootstrap.

For each piece type (Q, R, B, N, P) and each colour (STM / opponent),
takes a realistic board position from the source dataset, removes one
piece of that type, evaluates with Stockfish at the given depth, and
keeps positions where the eval sign is consistent with the imbalance.

Output format is identical to reeval_stockfish.py — train/val split,
tensors (N,14,8,8) uint8, values float32, move_idxs int64, legal_masks
(N,512) uint8 packed.

Usage:
    python3 gen_material_imbalance.py \\
        --source  /blackhole/dataset_2021_06_mid_sf18.pt \\
        --out     /blackhole/dataset_material_imbalance.pt \\
        --stockfish ~/bin/stockfish \\
        --depth   18 \\
        --n-per-combo 20000 \\
        --workers 16 \\
        --seed    42
"""

import argparse
import math
import multiprocessing
import os
import random
import signal
import subprocess
import sys
import time

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board import board_to_tensor, move_to_index

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PIECE_TYPES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
PIECE_NAME  = {
    chess.QUEEN:  "queen",
    chess.ROOK:   "rook",
    chess.BISHOP: "bishop",
    chess.KNIGHT: "knight",
    chess.PAWN:   "pawn",
}
# Minimum |eval| to keep — positions below this are likely tactically
# complicated enough that the material imbalance isn't the dominant factor.
PIECE_MIN_EVAL = {
    chess.QUEEN:  0.55,
    chess.ROOK:   0.30,
    chess.BISHOP: 0.10,
    chess.KNIGHT: 0.10,
    chess.PAWN:   0.03,
}
# "STM" = remove from side to move (STM loses piece → eval should be negative)
# "OPP" = remove from opponent   (STM gains piece  → eval should be positive)
SIDES = ["stm", "opp"]

VAL_FRACTION = 0.05   # 5% validation split

# ---------------------------------------------------------------------------
# Stockfish interface
# ---------------------------------------------------------------------------

class Stockfish:
    def __init__(self, path, depth, threads=1):
        self.depth = depth
        self._proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self._send("uci");           self._wait("uciok")
        self._send("isready");       self._wait("readyok")
        self._send(f"setoption name Threads value {threads}")

    def _send(self, cmd):
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _wait(self, token):
        while True:
            line = self._proc.stdout.readline().strip()
            if token in line:
                return

    def evaluate(self, fen):
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        cp = mate = None
        bestmove = "(none)"
        for line in iter(self._proc.stdout.readline, ""):
            line = line.strip()
            if line.startswith("info") and "score" in line:
                parts = line.split()
                try:
                    si = parts.index("score")
                    t, v = parts[si+1], int(parts[si+2])
                    if t == "cp":   cp   = v
                    elif t == "mate": mate = v
                except (ValueError, IndexError):
                    pass
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) > 1:
                    bestmove = parts[1]
                break
        if mate is not None:
            value = 1.0 if mate > 0 else -1.0
        elif cp is not None:
            value = math.tanh(cp / 400.0)
        else:
            return None, None
        return value, bestmove

    def close(self):
        try:
            self._send("quit")
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------

def _remove_piece(fen: str, piece_type: int, side: str) -> str | None:
    """
    Remove one random piece of piece_type from the specified side.
    side: "stm" (side to move) or "opp" (the other side).
    Returns modified FEN, or None if no such piece exists or result is invalid.
    """
    board = chess.Board(fen)
    color = board.turn if side == "stm" else (not board.turn)

    squares = list(board.pieces(piece_type, color))
    if not squares:
        return None

    sq = random.choice(squares)
    board.remove_piece_at(sq)

    # Clear castling rights that reference the removed square if it was a rook
    # (python-chess doesn't auto-update castling rights on manual piece removal)
    board.castling_rights &= ~chess.BB_SQUARES[sq]

    # Validity checks
    if board.is_check():          # revealing a check by removal is unrealistic
        return None
    if not list(board.legal_moves):
        return None               # stalemate or no legal moves

    return board.fen()


def _worker_init(sf_path, depth):
    global _sf
    _sf = Stockfish(sf_path, depth)

    def _on_sigterm(signum, frame):
        try:
            _sf.close()
        except Exception:
            pass
        os._exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)


def _worker_eval(args):
    """Evaluate one (source_fen, piece_type, side) tuple."""
    source_fen, piece_type, side = args
    mod_fen = _remove_piece(source_fen, piece_type, side)
    if mod_fen is None:
        return None

    value, bestmove = _sf.evaluate(mod_fen)
    if value is None:
        return None

    # Sign check: removing STM piece → value should be negative; OPP → positive
    expected_sign = -1 if side == "stm" else +1
    if value * expected_sign < PIECE_MIN_EVAL[piece_type]:
        return None

    # Encode board
    board = chess.Board(mod_fen)
    flip  = (board.turn == chess.BLACK)

    tensor = board_to_tensor(board).numpy().astype(np.uint8)

    if bestmove and bestmove != "(none)":
        try:
            move     = chess.Move.from_uci(bestmove)
            move_idx = move_to_index(move, flip=flip)
        except Exception:
            move_idx = 0
    else:
        move_idx = 0

    # Legal move mask — 4096 bits → 512 bytes packed
    mask = np.zeros(4096, dtype=np.uint8)
    for m in board.legal_moves:
        mask[move_to_index(m, flip=flip)] = 1
    packed = np.packbits(mask)

    return tensor, np.float32(value), np.int64(move_idx), packed


# ---------------------------------------------------------------------------
# Per-combo generation
# ---------------------------------------------------------------------------

def generate_combo(pool, source_fens, piece_type, side, n_target, rng):
    """Generate up to n_target valid positions for (piece_type, side)."""
    name = f"{PIECE_NAME[piece_type]}/{side}"
    print(f"  [{name}]  target={n_target:,}  source pool={len(source_fens):,}")

    idxs = list(range(len(source_fens)))
    rng.shuffle(idxs)
    tasks = [(source_fens[i], piece_type, side) for i in idxs]

    tensors, values, move_idxs, masks = [], [], [], []
    t0 = time.time()

    # Reuse the shared pool — no per-combo pool creation/termination.
    # imap_unordered streams results; we stop consuming once n_target reached.
    it = pool.imap_unordered(_worker_eval, tasks, chunksize=64)
    try:
        for result in it:
            if result is None:
                continue
            t, v, mi, pk = result
            tensors.append(t)
            values.append(v)
            move_idxs.append(mi)
            masks.append(pk)

            done = len(tensors)
            if done % 1000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                print(f"    {done:,} / {n_target:,}  ({rate:.1f} pos/s)")
            if done >= n_target:
                break
    finally:
        it._coro = None   # discard the iterator without draining remaining tasks

    n = len(tensors)
    elapsed = time.time() - t0
    print(f"  [{name}]  done: {n:,} positions in {elapsed:.0f}s")
    if n == 0:
        return None
    return (
        np.stack(tensors),
        np.array(values, dtype=np.float32),
        np.array(move_idxs, dtype=np.int64),
        np.stack(masks),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",      required=True,
                    help="Source dataset (.pt) — FENs are taken from here")
    ap.add_argument("--out",         required=True,
                    help="Output dataset path")
    ap.add_argument("--stockfish",   default="/usr/games/stockfish")
    ap.add_argument("--depth",       type=int, default=18)
    ap.add_argument("--n-per-combo", type=int, default=20000,
                    help="Target positions per (piece_type × side) combination")
    ap.add_argument("--workers",     type=int, default=16)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load source FENs (combine train + val)
    print(f"Loading source dataset: {args.source}")
    src = torch.load(args.source, map_location="cpu", weights_only=False)
    fens = src["train"].get("fens", []) + src["val"].get("fens", [])
    print(f"  Source FENs: {len(fens):,}")
    if not fens:
        print("ERROR: source dataset has no FENs stored.")
        sys.exit(1)

    # Single pool for all combos — avoids pool recreation/termination bugs.
    all_tensors, all_values, all_midxs, all_masks = [], [], [], []

    with multiprocessing.Pool(args.workers, initializer=_worker_init,
                              initargs=(args.stockfish, args.depth)) as pool:
        for piece_type in PIECE_TYPES:
            for side in SIDES:
                result = generate_combo(
                    pool, fens, piece_type, side,
                    n_target=args.n_per_combo,
                    rng=rng,
                )
                if result is None:
                    continue
                t, v, mi, pk = result
                all_tensors.append(t)
                all_values.append(v)
                all_midxs.append(mi)
                all_masks.append(pk)

    # Concatenate
    tensors   = np.concatenate(all_tensors)
    values    = np.concatenate(all_values)
    move_idxs = np.concatenate(all_midxs)
    masks     = np.concatenate(all_masks)
    N = len(values)
    print(f"\nTotal positions: {N:,}")

    # Shuffle and split
    idx = np.random.default_rng(args.seed).permutation(N)
    n_val   = max(1, int(N * VAL_FRACTION))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    def pack(subset):
        return {
            "tensors":    torch.from_numpy(tensors[subset]),
            "values":     torch.from_numpy(values[subset]),
            "move_idxs":  torch.from_numpy(move_idxs[subset]),
            "legal_masks":torch.from_numpy(masks[subset]),
        }

    out = {
        "train": pack(tr_idx),
        "val":   pack(val_idx),
        "meta":  {
            "source":    args.source,
            "depth":     args.depth,
            "n_per_combo": args.n_per_combo,
            "n_combos":  len(PIECE_TYPES) * len(SIDES),
            "n_train":   len(tr_idx),
            "n_val":     n_val,
        },
    }

    # Value distribution summary
    v_all = values
    print(f"\nValue distribution:")
    print(f"  Mean: {v_all.mean():+.4f}   Std: {v_all.std():.4f}")
    for label, lo, hi in [
        ("≥0.7 decisive win",  0.7,  1.1),
        ("0.5–0.7",            0.5,  0.7),
        ("0.1–0.5 mild win",   0.1,  0.5),
        ("|v|<0.1 equal",     -0.1,  0.1),
        ("-0.5–-0.1 mild loss",-0.5, -0.1),
        ("-0.7–-0.5",          -0.7, -0.5),
        ("≤-0.7 decisive loss",-1.1, -0.7),
    ]:
        pct = ((v_all >= lo) & (v_all < hi)).mean() * 100
        print(f"  {label}: {pct:.1f}%")

    torch.save(out, args.out)
    print(f"\nSaved → {args.out}")
    print(f"  train: {len(tr_idx):,}  val: {n_val:,}")


if __name__ == "__main__":
    main()
