"""
gen_near_mate.py — Generate positions with forced mate in ≤ max_mate moves.

Sources KQ vs K (stage 1), KR vs K (stage 2), and KQ vs KR (stage 4)
as candidate positions, evaluates each with Stockfish depth-18, and
keeps only positions where a forced mate is confirmed within max_mate
half-moves from either side.

Labels: +1.0 if the side to move delivers the mate, -1.0 if the side
to move is being mated.

Usage:
    python3 gen_near_mate.py \\
        --out     /blackhole/dataset_near_mate.pt \\
        --stockfish ~/bin/stockfish \\
        --n       220000 \\
        --max-mate 7 \\
        --workers  16
"""

import argparse, math, multiprocessing, os, random, signal, subprocess, sys, time
import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board import board_to_tensor, move_to_index
from generate_endgame import generate_positions

VAL_FRACTION = 0.05
_sf = None


# ---------------------------------------------------------------------------
# Stockfish
# ---------------------------------------------------------------------------

class Stockfish:
    def __init__(self, path, depth, threads=1):
        self.depth = depth
        self._proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self._send("uci");         self._wait("uciok")
        self._send("isready");     self._wait("readyok")
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
        """Returns (value, mate_in, bestmove) or (None, None, None)."""
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
                    if t == "cp":     cp   = v
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
            return value, mate, bestmove
        elif cp is not None:
            return math.tanh(cp / 400.0), None, bestmove
        return None, None, None

    def close(self):
        try:
            self._send("quit")
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

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
    fen, max_mate = args
    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over():
        return None

    value, mate_in, bestmove = _sf.evaluate(fen)
    if value is None or mate_in is None:
        return None
    if abs(mate_in) > max_mate:
        return None

    flip     = (board.turn == chess.BLACK)
    tensor   = board_to_tensor(board).numpy().astype(np.uint8)

    if bestmove and bestmove != "(none)":
        try:
            move_idx = move_to_index(chess.Move.from_uci(bestmove), flip=flip)
        except Exception:
            move_idx = 0
    else:
        move_idx = 0

    mask = np.zeros(4096, dtype=np.uint8)
    for m in board.legal_moves:
        mask[move_to_index(m, flip=flip)] = 1
    packed = np.packbits(mask)

    return tensor, np.float32(value), np.int64(move_idx), packed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",       required=True)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--depth",     type=int, default=18)
    ap.add_argument("--n",         type=int, default=220000,
                    help="Target number of near-mate positions")
    ap.add_argument("--max-mate",  type=int, default=7,
                    help="Keep positions with forced mate in ≤ this many moves")
    ap.add_argument("--workers",   type=int, default=16)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Generate candidate positions from decisive endgame stages
    print(f"Generating candidate positions (stages 1, 2, 4) ...")
    # Over-generate: only a fraction will pass the mate filter
    n_candidates = args.n * 6
    candidates = generate_positions(n_candidates, stages=[1, 2, 4], include_mirrors=True)
    fens = [item[0].fen() if hasattr(item[0], 'fen') else item[0] for item in candidates]
    rng.shuffle(fens)
    print(f"  Candidates: {len(fens):,}")

    tensors, values, move_idxs, masks = [], [], [], []
    t0 = time.time()

    print(f"\nFiltering for mate-in-{args.max_mate} with SF depth {args.depth} ...")
    tasks = [(fen, args.max_mate) for fen in fens]

    with multiprocessing.Pool(args.workers, initializer=_worker_init,
                               initargs=(args.stockfish, args.depth)) as pool:
        for result in pool.imap_unordered(_worker_eval, tasks):
            if result is None:
                continue
            t, v, mi, pk = result
            tensors.append(t)
            values.append(v)
            move_idxs.append(mi)
            masks.append(pk)

            done = len(tensors)
            if done % 5000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                scanned = done + sum(1 for _ in [])   # approximate
                print(f"  {done:,} / {args.n:,}  ({rate:.1f} pos/s)")
            if done >= args.n:
                pool.terminate()
                break

    n = len(tensors)
    print(f"\nCollected {n:,} near-mate positions (mate ≤ {args.max_mate})")

    # Train/val split
    idxs = list(range(n))
    rng.shuffle(idxs)
    n_val   = max(1, int(n * VAL_FRACTION))
    val_i   = idxs[:n_val]
    train_i = idxs[n_val:]

    def _split(subset):
        t = np.stack([tensors[i]   for i in subset])
        v = np.array([values[i]    for i in subset], dtype=np.float32)
        m = np.array([move_idxs[i] for i in subset], dtype=np.int64)
        p = np.stack([masks[i]     for i in subset])
        wins  = (v > 0.5).sum()
        draws = (np.abs(v) <= 0.5).sum()
        losses= (v < -0.5).sum()
        print(f"    n={len(subset):,}  win={wins}  draw={draws}  loss={losses}")
        return {
            "tensors":     torch.from_numpy(t),
            "values":      torch.from_numpy(v),
            "move_idxs":   torch.from_numpy(m),
            "legal_masks": torch.from_numpy(p),
            "drawness_mask":    torch.zeros(len(subset), dtype=torch.bool),
            "drawness_targets": torch.zeros(len(subset), dtype=torch.float32),
            "drawness_available": torch.ones(len(subset), dtype=torch.bool),
        }

    print("Building dataset ...")
    print("  train:")
    train_split = _split(train_i)
    print("  val:")
    val_split   = _split(val_i)

    out = {
        "train": train_split,
        "val":   val_split,
        "meta":  {"source": "near_mate", "max_mate": args.max_mate,
                  "depth": args.depth, "n": n},
    }
    torch.save(out, args.out)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
