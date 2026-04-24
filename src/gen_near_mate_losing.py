"""
gen_near_mate_losing.py — Generate 'being mated' positions for near-mate balance.

For each winning near-mate position (STM mates in ≤ max_mate_win), plays SF's
first mating move to land in a position where the LOSING side is now to move.
That position is encoded from the loser's perspective (value = -1.0) with SF's
best defensive response as the policy target.

Also collects naturally-occurring 'being mated' positions from the source pool
(where SF already reports mate_in < 0 and |mate_in| ≤ max_mate_lose).

Save to a separate file; merge with the existing near-mate dataset before training.

Expected yield from the 310k decisive-FEN pool (low+mid+high SF-18):
  ~52k derived  (from mate-in-2,3 winning positions; mate-in-1 produces checkmate
                 on the derived board and is skipped)
  ~15-25k natural (mate_in ≤ -1, naturally rare in real games)
  Total: ~65-77k.  Use --max-mate-win 5 to push closer to 100k.

Usage:
    python3 gen_near_mate_losing.py \\
        --sources /blackhole/dataset_2021_06_{low,mid,high}_sf18.pt \\
        --out     /blackhole/dataset_near_mate_losing.pt \\
        --stockfish ~/bin/stockfish \\
        --depth 18 --n 100000 --workers 16
"""

import argparse, math, multiprocessing, os, random, signal, subprocess, sys, time
import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board import board_to_tensor, move_to_index

VAL_FRACTION       = 0.05
PREFILTER_THRESHOLD = 0.90
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
            return 1.0 if mate > 0 else -1.0, mate, bestmove
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


def _encode(board, value, bestmove):
    flip = (board.turn == chess.BLACK)
    tensor = board_to_tensor(board).numpy().astype(np.uint8)
    move_idx = 0
    if bestmove and bestmove != "(none)":
        try:
            move_idx = move_to_index(chess.Move.from_uci(bestmove), flip=flip)
        except Exception:
            pass
    mask = np.zeros(4096, dtype=np.uint8)
    for m in board.legal_moves:
        mask[move_to_index(m, flip=flip)] = 1
    return tensor, np.float32(value), np.int64(move_idx), np.packbits(mask)


def _worker_eval(args):
    fen, max_mate_win, max_mate_lose = args
    try:
        board = chess.Board(fen)
    except Exception:
        return None
    if not board.is_valid() or board.is_game_over():
        return None

    value, mate_in, bestmove = _sf.evaluate(fen)
    if value is None or mate_in is None:
        return None

    # Path 1: STM is winning — push the first mating move to derive a position
    # where the losing side is to move and is being mated.
    # Skip mate-in-1: the derived board would be immediate checkmate (unusable).
    if 1 < mate_in <= max_mate_win:
        if not bestmove or bestmove == "(none)":
            return None
        try:
            board.push(chess.Move.from_uci(bestmove))
        except Exception:
            return None
        if board.is_game_over():
            return None
        # Second SF call: get the best defensive response in the losing position.
        value2, _, bestmove2 = _sf.evaluate(board.fen())
        if value2 is None:
            return None
        return _encode(board, -1.0, bestmove2)

    # Path 2: STM is already the losing side — collect directly.
    if -max_mate_lose <= mate_in < 0:
        return _encode(board, -1.0, bestmove)

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources",       nargs="+", required=True)
    ap.add_argument("--out",           required=True)
    ap.add_argument("--stockfish",     default="/usr/games/stockfish")
    ap.add_argument("--depth",         type=int, default=18)
    ap.add_argument("--n",             type=int, default=100000,
                    help="Target number of losing positions")
    ap.add_argument("--max-mate-win",  type=int, default=3,
                    help="Derive losing from winning positions with mate ≤ this "
                         "(use 5 to push closer to 100k from the 310k pool)")
    ap.add_argument("--max-mate-lose", type=int, default=5,
                    help="Also collect natural losing positions with |mate| ≤ this")
    ap.add_argument("--workers",       type=int, default=16)
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    all_fens = []
    for path in args.sources:
        print(f"Loading {path} ...")
        ds = torch.load(path, map_location="cpu", weights_only=False)
        for split in ("train", "val"):
            if split not in ds:
                continue
            fens   = ds[split].get("fens", [])
            values = ds[split].get("values", None)
            if not fens:
                continue
            if values is not None:
                vals = values.numpy() if hasattr(values, "numpy") else values
                keep = [f for f, v in zip(fens, vals) if abs(float(v)) >= PREFILTER_THRESHOLD]
            else:
                keep = fens
            all_fens.extend(keep)
        print(f"  Cumulative after pre-filter (|v| ≥ {PREFILTER_THRESHOLD}): {len(all_fens):,} FENs")

    if not all_fens:
        print("ERROR: no FENs found.")
        sys.exit(1)

    rng.shuffle(all_fens)
    print(f"\nTotal candidates: {len(all_fens):,}")
    print(f"Deriving losing positions (target: {args.n:,}) ...")
    print(f"  Derived path:  winning mate ≤ {args.max_mate_win} → push first move")
    print(f"  Natural path:  losing  |mate| ≤ {args.max_mate_lose}")

    tasks = [(fen, args.max_mate_win, args.max_mate_lose) for fen in all_fens]
    tensors, values, move_idxs, masks = [], [], [], []
    t0 = time.time()

    with multiprocessing.Pool(args.workers, initializer=_worker_init,
                              initargs=(args.stockfish, args.depth)) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker_eval, tasks)):
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
                rate = (i + 1) / elapsed
                print(f"  {done:,} collected  (scanned {i+1:,} at {rate:.1f} pos/s)")
            if done >= args.n:
                pool.terminate()
                break

    n = len(tensors)
    print(f"\nCollected {n:,} losing near-mate positions")
    if n < args.n:
        print(f"WARNING: pool exhausted — only {n:,} of {args.n:,} target")
        print(f"  Try --max-mate-win 5 to increase derived yield")

    idxs = list(range(n))
    rng.shuffle(idxs)
    n_val   = max(1, int(n * VAL_FRACTION))
    val_i   = idxs[:n_val]
    train_i = idxs[n_val:]

    def _split(subset, label):
        print(f"    {label}: n={len(subset):,}  (all loss)")
        return {
            "tensors":             torch.from_numpy(np.stack([tensors[i]   for i in subset])),
            "values":              torch.from_numpy(np.array([values[i]    for i in subset], dtype=np.float32)),
            "move_idxs":           torch.from_numpy(np.array([move_idxs[i] for i in subset], dtype=np.int64)),
            "legal_masks":         torch.from_numpy(np.stack([masks[i]     for i in subset])),
            "drawness_mask":       torch.zeros(len(subset), dtype=torch.bool),
            "drawness_targets":    torch.zeros(len(subset), dtype=torch.float32),
            "drawness_available":  torch.ones(len(subset), dtype=torch.bool),
        }

    print("Building dataset ...")
    train_split = _split(train_i, "train")
    val_split   = _split(val_i,   "val")

    torch.save({
        "train": train_split,
        "val":   val_split,
        "meta":  {
            "source":        "near_mate_losing",
            "max_mate_win":  args.max_mate_win,
            "max_mate_lose": args.max_mate_lose,
            "depth":         args.depth,
            "n":             n,
        },
    }, args.out)
    print(f"\nSaved → {args.out}")
    print(f"  train: {len(train_i):,}  val: {n_val:,}")


if __name__ == "__main__":
    main()
