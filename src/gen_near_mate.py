"""
gen_near_mate.py — Generate positions with forced mate in ≤ max_mate moves.

Loads FENs from one or more real game datasets (mid-band, low-band, etc.),
pre-filters to positions with high decisive value (likely forced outcomes),
then confirms with Stockfish depth-18 that a forced mate exists within
max_mate moves. Covers all position types (opening, middlegame, endgame)
rather than constructed endgame stages.

Labels: +1.0 if the side to move delivers the mate, -1.0 if mated.

Usage:
    python3 gen_near_mate.py \\
        --sources  /blackhole/dataset_2021_06_mid_sf18.pt \\
                   /blackhole/dataset_2021_06_low_sf18.pt \\
        --out      /blackhole/dataset_near_mate.pt \\
        --stockfish ~/bin/stockfish \\
        --n        50000 \\
        --max-mate 3 \\
        --workers  16
"""

import argparse, math, multiprocessing, os, random, signal, subprocess, sys, time
import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board import board_to_tensor, move_to_index

VAL_FRACTION = 0.05
_sf = None

# Pre-filter threshold: only re-evaluate positions where existing SF value
# is already decisive. tanh(cp/400) > 0.90 ≈ cp > 940. This is conservative —
# mate-in-3 positions will typically have |value| close to 1.0 already.
PREFILTER_THRESHOLD = 0.90


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
        """Returns (value, mate_in, bestmove) or (None, None, None).
        mate_in is None for centipawn scores."""
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


def _worker_eval(args):
    fen, max_mate = args
    try:
        board = chess.Board(fen)
    except Exception:
        return None
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
    ap.add_argument("--sources",   nargs="+", required=True,
                    help="Source dataset(s) — FENs are pooled from all")
    ap.add_argument("--out",       required=True)
    ap.add_argument("--stockfish", default="/usr/games/stockfish")
    ap.add_argument("--depth",     type=int, default=18)
    ap.add_argument("--n",         type=int, default=50000,
                    help="Target number of near-mate positions")
    ap.add_argument("--max-mate",  type=int, default=3,
                    help="Keep positions with forced mate in ≤ this many moves")
    ap.add_argument("--workers",   type=int, default=16)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load FENs from all source datasets, pre-filter by decisive value
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
                print(f"  WARNING: no FENs in {split} split of {path}")
                continue
            if values is not None:
                vals = values.numpy() if hasattr(values, "numpy") else values
                keep = [f for f, v in zip(fens, vals) if abs(float(v)) >= PREFILTER_THRESHOLD]
            else:
                keep = fens
            all_fens.extend(keep)
        print(f"  After pre-filter (|v| ≥ {PREFILTER_THRESHOLD}): {len(all_fens):,} FENs")

    if not all_fens:
        print("ERROR: no FENs found in source datasets.")
        sys.exit(1)

    rng.shuffle(all_fens)
    print(f"\nTotal candidate FENs: {len(all_fens):,}")
    print(f"Filtering for mate-in-{args.max_mate} with SF depth {args.depth} ...")

    tensors, values, move_idxs, masks = [], [], [], []
    t0 = time.time()
    tasks = [(fen, args.max_mate) for fen in all_fens]

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
    print(f"\nCollected {n:,} near-mate positions (mate ≤ {args.max_mate})")
    if n < args.n:
        print(f"WARNING: only {n:,} collected — source pool exhausted before reaching {args.n:,}")

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
        wins   = int((v >  0.5).sum())
        losses = int((v < -0.5).sum())
        print(f"    n={len(subset):,}  win={wins:,}  loss={losses:,}")
        return {
            "tensors":              torch.from_numpy(t),
            "values":               torch.from_numpy(v),
            "move_idxs":            torch.from_numpy(m),
            "legal_masks":          torch.from_numpy(p),
            "drawness_mask":        torch.zeros(len(subset), dtype=torch.bool),
            "drawness_targets":     torch.zeros(len(subset), dtype=torch.float32),
            "drawness_available":   torch.ones(len(subset), dtype=torch.bool),
        }

    print("Building dataset ...")
    print("  train:"); train_split = _split(train_i)
    print("  val:");   val_split   = _split(val_i)

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
