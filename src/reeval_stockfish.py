"""
Re-evaluate dataset positions with Stockfish.

Loads an existing dataset.pt, evaluates each position with Stockfish at a
fixed depth, replaces game-outcome labels with engine evaluations, and saves
a new dataset_sf.pt ready for training.

Label convention (matches training pipeline):
  Stockfish centipawn eval is already in the side-to-move perspective
  by UCI protocol — no flip is applied here. Squashed through
  tanh(cp / 400) → range (-1, +1).
  +1 = side to move is winning decisively
   0 = equal
  -1 = side to move is losing decisively

  Mate scores: mapped to ±1.0 directly.

  Note: "STM-relative" fixes in board.py and data.py apply to board tensor
  encoding and move indexing. They are unrelated to SF score convention.

Why tanh(cp / 400)?
  400cp is roughly one piece advantage. tanh saturates gracefully so a
  +800cp position doesn't dominate the MSE loss over a +200cp position.
  AlphaZero uses a similar squashing convention.

Usage — single job (original behaviour)
-----
    python3 src/reeval_stockfish.py --dataset dataset.pt --out dataset_sf.pt
    python3 src/reeval_stockfish.py --dataset dataset.pt --out dataset_sf.pt \\
        --depth 12 --n 200000 --stockfish /usr/games/stockfish --workers 32

Usage — chunked reeval (split across multiple HPC jobs, each ≤ 24h wall)
-----
    # Job 0 of 3:
    python3 src/reeval_stockfish.py --dataset dataset.pt \\
        --n 400000 --seed 42 --depth 20 --workers 16 \\
        --stockfish /path/to/stockfish \\
        --chunk-idx 0 --n-chunks 3 --partial-out /scratch/part_0.pt

    # Job 1 of 3  (same --n, --seed as above):
    python3 src/reeval_stockfish.py --dataset dataset.pt \\
        --n 400000 --seed 42 --depth 20 --workers 16 \\
        --stockfish /path/to/stockfish \\
        --chunk-idx 1 --n-chunks 3 --partial-out /scratch/part_1.pt

    # Job 2 of 3:
    python3 src/reeval_stockfish.py --dataset dataset.pt \\
        --n 400000 --seed 42 --depth 20 --workers 16 \\
        --stockfish /path/to/stockfish \\
        --chunk-idx 2 --n-chunks 3 --partial-out /scratch/part_2.pt

    # Merge (no SF/workers needed — just loads partials and applies filters):
    python3 src/reeval_stockfish.py --dataset dataset.pt --out dataset_sf.pt \\
        --merge /scratch/part_0.pt /scratch/part_1.pt /scratch/part_2.pt \\
        --min-decisive 0.0 --max-pieces 32

    --n, --seed must be identical across all chunk jobs and the merge call.
    --chunk-idx and --n-chunks determine which slice of sampled positions each
    job evaluates. All other args (depth, workers, stockfish) are per-chunk only.
"""

import argparse
import math
import multiprocessing
import os
import signal
import subprocess
import sys
import time

import chess
import numpy as np
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

        # UCI protocol guarantees score cp/mate is always from the side-to-move's
        # perspective — Stockfish flips internally. No Python-level flip is needed
        # or correct here. "STM-relative" fixes elsewhere (board.py, data.py) are
        # about board tensor encoding and move indexing, not SF score convention.
        if mate is not None:
            return (1.0 if mate > 0 else -1.0), bestmove
        if cp is not None:
            return math.tanh(cp / 400.0), bestmove
        return 0.0, bestmove

    def close(self):
        self._send("quit")
        self._proc.wait()


# ---------------------------------------------------------------------------
# Early-stop flag — set by SIGTERM/SIGINT handler so the eval loop can save
# a checkpoint before the OS sends SIGKILL.
# ---------------------------------------------------------------------------

_stop_early = False

def _handle_sigterm(signum, frame):
    global _stop_early
    print(f"\nSignal {signum} received — will checkpoint after current batch.",
          flush=True)
    _stop_early = True


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

        # Legal move mask bit-packed: 4096 bits → 512 bytes per position.
        # Use numpy.packbits (stable since NumPy 1.x) instead of
        # torch.packbits which was only added in PyTorch 1.7.0.
        mask = np.zeros(4096, dtype=np.uint8)
        for m in board.legal_moves:
            mask[move_to_index(m, flip=flip)] = 1
        packed_mask = torch.from_numpy(np.packbits(mask))   # shape (512,)

        return idx, val, sf_move_idx, packed_mask, False
    except Exception as e:
        # Surface the exception type on first few errors so silent bugs
        # like a missing torch API are diagnosable.
        if not hasattr(_eval_one, "_err_reported"):
            _eval_one._err_reported = 0
        if _eval_one._err_reported < 3:
            print(f"[_eval_one error #{_eval_one._err_reported+1}] {type(e).__name__}: {e}",
                  flush=True)
            _eval_one._err_reported += 1
        return idx, 0.0, -1, None, True


# ---------------------------------------------------------------------------
# Shared helpers for single-job and chunked modes
# ---------------------------------------------------------------------------

def _load_and_sample(dataset_path: str, n: int, seed: int):
    """
    Load dataset, merge train+val, subsample n positions with fixed seed.

    Returns a dict with everything downstream needs:
      sampled_idxs  : list[int]       indices into the merged pool
      all_fens      : list[str]
      all_tensors   : Tensor (N, ...)
      all_moves     : Tensor (N,)
      all_visit_dists: Tensor or None
      n_train_orig  : int             boundary for restoring train/val split
      has_vd        : bool
    """
    import random as _random
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)

    train_d      = data["train"]
    val_d        = data["val"]
    n_train_orig = len(train_d["tensors"])

    all_tensors = torch.cat([train_d["tensors"],  val_d["tensors"]],  dim=0)
    all_fens    = train_d["fens"] + val_d["fens"]
    all_moves   = torch.cat([train_d["move_idxs"], val_d["move_idxs"]], dim=0)

    def _cat_optional(name):
        if name in train_d and name in val_d:
            return torch.cat([train_d[name], val_d[name]], dim=0)
        return None

    all_outcome_values = _cat_optional("outcome_values")
    all_game_ids       = _cat_optional("game_ids")
    all_plys           = _cat_optional("plys")
    all_drawness_mask  = _cat_optional("drawness_mask")
    all_drawness_targets = _cat_optional("drawness_targets")
    all_drawness_available = _cat_optional("drawness_available")

    has_vd = "visit_dists" in train_d
    if has_vd:
        all_visit_dists = torch.cat([train_d["visit_dists"], val_d["visit_dists"]], dim=0)
    else:
        all_visit_dists = None

    del data, train_d, val_d
    import gc; gc.collect()

    total = len(all_fens)
    if n is not None and n < total:
        rng          = _random.Random(seed)
        sampled_idxs = sorted(rng.sample(range(total), n))
        all_tensors  = all_tensors[sampled_idxs]
        all_fens     = [all_fens[i] for i in sampled_idxs]
        all_moves    = all_moves[sampled_idxs]
        if all_visit_dists is not None:
            all_visit_dists = all_visit_dists[sampled_idxs]
        if all_outcome_values is not None:
            all_outcome_values = all_outcome_values[sampled_idxs]
        if all_game_ids is not None:
            all_game_ids = all_game_ids[sampled_idxs]
        if all_plys is not None:
            all_plys = all_plys[sampled_idxs]
        if all_drawness_mask is not None:
            all_drawness_mask = all_drawness_mask[sampled_idxs]
        if all_drawness_targets is not None:
            all_drawness_targets = all_drawness_targets[sampled_idxs]
        if all_drawness_available is not None:
            all_drawness_available = all_drawness_available[sampled_idxs]
    else:
        sampled_idxs = list(range(total))

    return {
        "sampled_idxs":   sampled_idxs,
        "all_fens":       all_fens,
        "all_tensors":    all_tensors,
        "all_moves":      all_moves,
        "all_visit_dists": all_visit_dists,
        "all_outcome_values": all_outcome_values,
        "all_game_ids":   all_game_ids,
        "all_plys":       all_plys,
        "all_drawness_mask": all_drawness_mask,
        "all_drawness_targets": all_drawness_targets,
        "all_drawness_available": all_drawness_available,
        "n_train_orig":   n_train_orig,
        "has_vd":         has_vd,
    }


def _run_sf_on_slice(fens_slice, stockfish_path, depth, workers,
                     on_checkpoint=None, checkpoint_interval=10_000):
    """
    Evaluate a list of FEN strings with Stockfish.

    Returns (new_values, new_move_idxs, all_packed_masks, valid_mask, done)
    where done <= len(fens_slice) — may be less if stopped early by SIGTERM.

    on_checkpoint(done, values, move_idxs, masks, valid): called every
    checkpoint_interval positions and on early stop. Use this to flush a
    partial save so wall-time kills don't lose all progress.
    """
    n_slice = len(fens_slice)
    new_values       = torch.zeros(n_slice, dtype=torch.float32)
    new_move_idxs    = torch.full((n_slice,), -1, dtype=torch.long)
    all_packed_masks = torch.zeros(n_slice, 512, dtype=torch.uint8)
    valid_mask       = torch.zeros(n_slice, dtype=torch.bool)

    errors = sf_move_hits = done = 0
    t0 = time.time()

    with multiprocessing.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(stockfish_path, depth),
    ) as pool:
        print(f"  Workers ready ({workers} processes, depth={depth}).")
        for local_idx, val, sf_move_idx, packed_mask, err in pool.imap(
                _eval_one, enumerate(fens_slice), chunksize=64):
            new_values[local_idx] = val
            if not err:
                if sf_move_idx >= 0:
                    new_move_idxs[local_idx] = sf_move_idx
                    sf_move_hits += 1
                if packed_mask is not None:
                    all_packed_masks[local_idx] = packed_mask
                    valid_mask[local_idx] = True
            else:
                errors += 1
            done += 1
            if done % 5000 == 0:
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (n_slice - done) / rate
                print(f"  {done:>8,} / {n_slice:,}  "
                      f"({rate:.0f} pos/s)  ETA {eta/60:.0f} min", flush=True)
            if on_checkpoint and done % checkpoint_interval == 0:
                on_checkpoint(done, new_values, new_move_idxs, all_packed_masks, valid_mask)
            if _stop_early:
                print(f"  Stopping early at {done:,} / {n_slice:,}.", flush=True)
                if on_checkpoint:
                    on_checkpoint(done, new_values, new_move_idxs, all_packed_masks, valid_mask)
                pool.terminate()
                break

    elapsed = time.time() - t0
    print(f"\nDone: {done:,} positions in {elapsed:.0f}s  ({done/max(elapsed,1):.0f} pos/s)")
    if errors:
        print(f"WARNING: {errors} evaluation errors — these positions will be dropped at save time")
    print(f"SF best moves used: {sf_move_hits:,} / {done:,}  "
          f"(fallback to human moves: {done - sf_move_hits:,})")

    return new_values[:done], new_move_idxs[:done], all_packed_masks[:done], valid_mask[:done], done


def _apply_filter_and_save(
        dataset_state: dict,
        new_values: torch.Tensor,
        new_move_idxs_sf: torch.Tensor,
        all_packed_masks: torch.Tensor,
        valid_mask: torch.Tensor,
        out_path: str,
        min_decisive: float,
        max_pieces: int,
        derive_drawness_from_outcome: bool = False,
        drawness_sf_threshold: float = 0.15,
        drawness_min_ply: int = 40,
        drawness_target: float = 0.7,
        drawness_max_pieces: int = 32,
        drawness_game_level: bool = False,
        extra_meta: dict = None,
):
    """
    Apply decisive/piece-count filter, restore train/val split, save dataset.
    Shared between single-job reeval and merge.
    """
    all_fens        = dataset_state["all_fens"]
    all_tensors     = dataset_state["all_tensors"]
    all_moves_orig  = dataset_state["all_moves"]      # human-move fallback
    all_visit_dists = dataset_state["all_visit_dists"]
    all_outcome_values = dataset_state["all_outcome_values"]
    all_game_ids    = dataset_state["all_game_ids"]
    all_plys        = dataset_state["all_plys"]
    all_drawness_mask = dataset_state["all_drawness_mask"]
    all_drawness_targets = dataset_state["all_drawness_targets"]
    all_drawness_available = dataset_state["all_drawness_available"]
    sampled_idxs    = dataset_state["sampled_idxs"]
    n_train_orig    = dataset_state["n_train_orig"]
    has_vd          = dataset_state["has_vd"]

    # Merge SF move where available, fall back to human move
    final_move_idxs = all_moves_orig.clone()
    sf_hit_mask = new_move_idxs_sf >= 0
    final_move_idxs[sf_hit_mask] = new_move_idxs_sf[sf_hit_mask]

    # Drop positions where SF evaluation failed.
    # Failures are recorded as 0.0 in new_values, which would silently inject
    # fake equal-position labels into the dataset — especially harmful when
    # min_decisive=0.0 keeps all positions. Drop them unconditionally here
    # before any further filtering.
    n_bad = int((~valid_mask).sum().item())
    if n_bad > 0:
        print(f"\nDropping {n_bad} positions with evaluation errors "
              f"(would otherwise inject 0.0 labels).")
        keep_valid      = valid_mask.nonzero(as_tuple=True)[0]
        new_values      = new_values[keep_valid]
        all_tensors     = all_tensors[keep_valid]
        final_move_idxs = final_move_idxs[keep_valid]
        all_packed_masks= all_packed_masks[keep_valid]
        all_fens        = [all_fens[i] for i in keep_valid.tolist()]
        sampled_idxs    = [sampled_idxs[i] for i in keep_valid.tolist()]
        if all_visit_dists is not None:
            all_visit_dists = all_visit_dists[keep_valid]
        if all_outcome_values is not None:
            all_outcome_values = all_outcome_values[keep_valid]
        if all_game_ids is not None:
            all_game_ids = all_game_ids[keep_valid]
        if all_plys is not None:
            all_plys = all_plys[keep_valid]
        if all_drawness_mask is not None:
            all_drawness_mask = all_drawness_mask[keep_valid]
        if all_drawness_targets is not None:
            all_drawness_targets = all_drawness_targets[keep_valid]
        if all_drawness_available is not None:
            all_drawness_available = all_drawness_available[keep_valid]
        valid_mask = valid_mask[keep_valid]   # now all True

    # All remaining rows have valid evaluations and masks.
    has_masks = valid_mask.all().item()
    n = len(all_fens)

    # Print label distribution
    vals = new_values.numpy()
    print(f"\nLabel statistics (SF evals, tanh-squashed):")
    print(f"  Mean:   {vals.mean():.4f}")
    print(f"  Std:    {vals.std():.4f}")
    print(f"  Min:    {vals.min():.4f}")
    print(f"  Max:    {vals.max():.4f}")
    print(f"  |v|>0.5 (decisive): {(np.abs(vals) > 0.5).mean()*100:.1f}%")
    print(f"  |v|<0.1 (equal):    {(np.abs(vals) < 0.1).mean()*100:.1f}%")

    keep = list(range(n))
    if min_decisive > 0.0 or max_pieces < 32:
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
        print(f"  Kept:             {len(keep):,} / {n:,}")
        print(f"  Dropped (eval):   {skipped_eval:,}")
        print(f"  Dropped (pieces): {skipped_pieces:,}")

        if len(keep) < 100:
            raise ValueError(
                f"Filter too aggressive — only {len(keep)} positions remain. "
                f"Lower min_decisive or raise max_pieces."
            )

        keep_t           = torch.tensor(keep, dtype=torch.long)
        new_values       = new_values[keep_t]
        all_tensors      = all_tensors[keep_t]
        final_move_idxs  = final_move_idxs[keep_t]
        all_packed_masks = all_packed_masks[keep_t]
        valid_mask       = valid_mask[keep_t]
        all_fens         = [all_fens[i] for i in keep]
        sampled_idxs     = [sampled_idxs[i] for i in keep]
        if all_visit_dists is not None:
            all_visit_dists = all_visit_dists[keep_t]
        if all_outcome_values is not None:
            all_outcome_values = all_outcome_values[keep_t]
        if all_game_ids is not None:
            all_game_ids = all_game_ids[keep_t]
        if all_plys is not None:
            all_plys = all_plys[keep_t]
        if all_drawness_mask is not None:
            all_drawness_mask = all_drawness_mask[keep_t]
        if all_drawness_targets is not None:
            all_drawness_targets = all_drawness_targets[keep_t]
        if all_drawness_available is not None:
            all_drawness_available = all_drawness_available[keep_t]
        has_masks = valid_mask.all().item()

        filtered_vals = new_values.numpy()
        print(f"  Post-filter decisive rate: "
              f"{(np.abs(filtered_vals) > 0.5).mean()*100:.1f}%")

    n = len(all_fens)
    if all_drawness_mask is None:
        all_drawness_mask = torch.zeros(n, dtype=torch.bool)
    else:
        all_drawness_mask = all_drawness_mask.bool()
    if all_drawness_targets is None:
        all_drawness_targets = torch.zeros(n, dtype=torch.float32)
    else:
        all_drawness_targets = all_drawness_targets.float()
    if all_drawness_available is None:
        has_existing_drawness = "all_drawness_mask" in dataset_state and dataset_state["all_drawness_mask"] is not None
        all_drawness_available = torch.full((n,), has_existing_drawness, dtype=torch.bool)
    else:
        all_drawness_available = all_drawness_available.bool()

    n_drawness_from_outcome = 0
    if derive_drawness_from_outcome:
        if all_outcome_values is None or all_plys is None:
            raise ValueError(
                "--derive-drawness-from-outcome requires input datasets parsed "
                "with outcome_values and plys. Re-parse the PGN with the updated data.py."
            )

        if drawness_game_level:
            # Game-level: a game is a structural draw only if the SF eval
            # NEVER exceeded the threshold for any of its sampled positions.
            # Excludes games where someone had a big advantage but blundered
            # back to 0 — those are not structural draws.
            if all_game_ids is None:
                raise ValueError(
                    "--drawness-game-level requires game_ids in the dataset. "
                    "Re-parse the PGN with the updated data.py."
                )
            is_draw = all_outcome_values.float().abs() < 1e-4
            abs_vals = new_values.float().abs()
            game_ids_np = all_game_ids.numpy() if hasattr(all_game_ids, 'numpy') else all_game_ids

            # For each game, find max |SF eval| across all its positions
            from collections import defaultdict
            game_max_eval: dict = defaultdict(float)
            game_is_draw: dict  = defaultdict(bool)
            for i in range(n):
                gid = int(game_ids_np[i])
                game_is_draw[gid] = bool(is_draw[i].item())
                cur = float(abs_vals[i].item())
                if cur > game_max_eval[gid]:
                    game_max_eval[gid] = cur

            candidate = torch.zeros(n, dtype=torch.bool)
            for i in range(n):
                gid = int(game_ids_np[i])
                if game_is_draw[gid] and game_max_eval[gid] < drawness_sf_threshold:
                    candidate[i] = True

            print(f"\nDrawness derived from drawn games (game-level):")
            print(f"  outcome draw & max|SF| across all positions < {drawness_sf_threshold}")
        else:
            # Position-level: each position checked independently
            candidate = (
                (all_outcome_values.float().abs() < 1e-4) &
                (new_values.float().abs() < drawness_sf_threshold) &
                (all_plys.long() >= drawness_min_ply)
            )
            print(f"\nDrawness derived from drawn games (position-level):")
            print(f"  outcome draw & |SF|<{drawness_sf_threshold} & ply>={drawness_min_ply}"
                  f" & pieces<={drawness_max_pieces}")

        if drawness_max_pieces < 32:
            piece_ok = torch.tensor([
                bin(int(chess.Board(fen).occupied)).count("1") <= drawness_max_pieces
                for fen in all_fens
            ], dtype=torch.bool)
            candidate &= piece_ok

        n_drawness_from_outcome = int(candidate.sum().item())
        all_drawness_mask = all_drawness_mask | candidate
        all_drawness_targets[candidate] = torch.maximum(
            all_drawness_targets[candidate],
            torch.full((n_drawness_from_outcome,), drawness_target, dtype=torch.float32),
        )
        all_drawness_available = torch.ones(n, dtype=torch.bool)

        print(f"  positives: {n_drawness_from_outcome:,} / {n:,}  target={drawness_target:.2f}")

    # Restore train/val split
    train_idxs = [i for i, orig in enumerate(sampled_idxs) if orig < n_train_orig]
    val_idxs   = [i for i, orig in enumerate(sampled_idxs) if orig >= n_train_orig]
    if not val_idxs:
        n_val      = max(1, len(train_idxs) // 20)
        val_idxs   = train_idxs[-n_val:]
        train_idxs = train_idxs[:-n_val]

    def pack(subset_idxs):
        d = {
            "tensors":   all_tensors[subset_idxs],
            "values":    new_values[subset_idxs],
            "move_idxs": final_move_idxs[subset_idxs],
            "fens":      [all_fens[i] for i in subset_idxs],
            "drawness_mask": all_drawness_mask[subset_idxs],
            "drawness_targets": all_drawness_targets[subset_idxs],
            "drawness_available": all_drawness_available[subset_idxs],
        }
        if all_outcome_values is not None:
            d["outcome_values"] = all_outcome_values[subset_idxs]
        if all_game_ids is not None:
            d["game_ids"] = all_game_ids[subset_idxs]
        if all_plys is not None:
            d["plys"] = all_plys[subset_idxs]
        if all_visit_dists is not None:
            d["visit_dists"] = all_visit_dists[subset_idxs]
        if has_masks:
            d["legal_masks"] = all_packed_masks[subset_idxs]
        return d

    meta = {
        "n_train":         len(train_idxs),
        "n_val":           len(val_idxs),
        "label_type":      "stockfish_tanh_cp400",
        "has_visit_dists": has_vd,
        "drawness_from_outcome": derive_drawness_from_outcome,
        "n_drawness_from_outcome": n_drawness_from_outcome,
    }
    if derive_drawness_from_outcome:
        meta.update({
            "drawness_sf_threshold": drawness_sf_threshold,
            "drawness_min_ply": drawness_min_ply,
            "drawness_target": drawness_target,
            "drawness_max_pieces": drawness_max_pieces,
        })
    if extra_meta:
        meta.update(extra_meta)

    out = {"train": pack(train_idxs), "val": pack(val_idxs), "meta": meta}
    torch.save(out, out_path)
    print(f"\nSaved → {out_path}")
    print(f"  train: {len(train_idxs):,}  val: {len(val_idxs):,}")
    if not has_masks:
        print(f"  legal_masks: NOT stored (some positions had evaluation errors)")


# ---------------------------------------------------------------------------
# Chunked reeval — evaluate one slice, save partial result
# ---------------------------------------------------------------------------

def reeval_chunk(dataset_path: str,
                 partial_out: str,
                 chunk_idx: int,
                 n_chunks: int,
                 stockfish_path: str,
                 depth: int,
                 n: int,
                 seed: int,
                 workers: int,
                 checkpoint_interval: int = 10_000):
    """
    Evaluate one chunk of positions and save a partial result file.

    All chunk jobs for the same dataset must use identical --n and --seed so
    that _load_and_sample produces the same sampled_idxs, making slices
    non-overlapping and covering the full set exactly.

    Checkpoints are flushed to partial_out every checkpoint_interval positions
    and on SIGTERM, so a wall-time kill preserves progress. The saved partial
    records the actual end index evaluated — the merge job will report any
    uncovered range so you can resubmit just that chunk.
    """
    print(f"=== Chunk {chunk_idx} / {n_chunks}  dataset={dataset_path} ===")
    state    = _load_and_sample(dataset_path, n, seed)
    n_total  = len(state["all_fens"])
    chunk_sz = (n_total + n_chunks - 1) // n_chunks   # ceiling division
    start    = chunk_idx * chunk_sz
    planned_end = min(start + chunk_sz, n_total)

    print(f"  Total positions: {n_total:,}  chunk size: {chunk_sz:,}")
    print(f"  This chunk: [{start:,}, {planned_end:,})  ({planned_end-start:,} positions)")

    fens_slice = state["all_fens"][start:planned_end]

    def _save(done_count, vals, moves, masks, valid):
        actual_end = start + done_count
        partial = {
            "chunk_idx":        chunk_idx,
            "n_chunks":         n_chunks,
            "n_total":          n_total,
            "n":                n,
            "seed":             seed,
            "depth":            depth,
            "start":            start,
            "end":              actual_end,
            "new_values":       vals[:done_count].clone(),
            "new_move_idxs":    moves[:done_count].clone(),
            "all_packed_masks": masks[:done_count].clone(),
            "valid_mask":       valid[:done_count].clone(),
        }
        torch.save(partial, partial_out)
        print(f"  Checkpoint → {partial_out}  [{start:,}, {actual_end:,})  ({done_count:,} positions)",
              flush=True)

    new_values, new_move_idxs, all_packed_masks, valid_mask, done = \
        _run_sf_on_slice(fens_slice, stockfish_path, depth, workers,
                         on_checkpoint=_save,
                         checkpoint_interval=checkpoint_interval)

    _save(done, new_values, new_move_idxs, all_packed_masks, valid_mask)
    actual_end = start + done
    print(f"\nPartial saved → {partial_out}")
    print(f"  chunk {chunk_idx}/{n_chunks}: positions [{start:,}, {actual_end:,})  ({done:,} evaluated)")
    if done < planned_end - start:
        print(f"  WARNING: chunk incomplete — {planned_end - actual_end:,} positions not evaluated.")
        print(f"  Merge will report the missing range. Resubmit CHUNK_IDX={chunk_idx} to complete.")


# ---------------------------------------------------------------------------
# Merge — combine partials into a final filtered dataset
# ---------------------------------------------------------------------------

def merge_partials(dataset_path: str,
                   partial_paths: list,
                   out_path: str,
                   min_decisive: float = 0.0,
                   max_pieces: int = 32,
                   n: int = 200_000,
                   seed: int = 42,
                   derive_drawness_from_outcome: bool = False,
                   drawness_sf_threshold: float = 0.15,
                   drawness_min_ply: int = 40,
                   drawness_target: float = 0.7,
                   drawness_max_pieces: int = 32,
                   drawness_game_level: bool = False):
    """
    Merge chunk partial files into a final dataset.

    n and seed must exactly match the values used when running chunk jobs.
    They are validated against the values recorded inside the partial files.
    """
    print(f"=== Merging {len(partial_paths)} partial files ===")

    # Load all partials and validate consistency
    partials = []
    seen_chunk_idxs = set()
    ref_n = ref_seed = ref_n_chunks = ref_n_total = ref_depth = None

    for path in sorted(partial_paths):
        p = torch.load(path, map_location="cpu", weights_only=False)

        if ref_n is None:
            ref_n        = p["n"]
            ref_seed     = p["seed"]
            ref_n_chunks = p["n_chunks"]
            ref_n_total  = p["n_total"]
            ref_depth    = p["depth"]
        else:
            errors = []
            if p["n"]        != ref_n:        errors.append(f"n={p['n']} (expected {ref_n})")
            if p["seed"]     != ref_seed:     errors.append(f"seed={p['seed']} (expected {ref_seed})")
            if p["n_chunks"] != ref_n_chunks: errors.append(f"n_chunks={p['n_chunks']} (expected {ref_n_chunks})")
            if p["n_total"]  != ref_n_total:  errors.append(f"n_total={p['n_total']} (expected {ref_n_total})")
            if p["depth"]    != ref_depth:    errors.append(f"depth={p['depth']} (expected {ref_depth})")
            if errors:
                raise ValueError(f"Partial {path} is inconsistent: {', '.join(errors)}")

        if p["chunk_idx"] in seen_chunk_idxs:
            raise ValueError(
                f"Duplicate chunk_idx={p['chunk_idx']} — "
                f"{path} covers the same chunk as a previously loaded partial."
            )
        seen_chunk_idxs.add(p["chunk_idx"])

        partials.append(p)
        print(f"  Loaded {path}  "
              f"chunk {p['chunk_idx']}/{p['n_chunks']}  "
              f"positions [{p['start']:,}, {p['end']:,})  depth={p['depth']}")

    # Validate caller's --n / --seed against what the chunks used.
    # argparse always supplies defaults so these are never None — validate
    # explicitly rather than silently overriding with partial values.
    if n != ref_n:
        raise ValueError(
            f"--n {n} does not match the value recorded in partials ({ref_n}). "
            f"Use --n {ref_n} to match the chunk jobs."
        )
    if seed != ref_seed:
        raise ValueError(
            f"--seed {seed} does not match the value recorded in partials ({ref_seed}). "
            f"Use --seed {ref_seed} to match the chunk jobs."
        )

    # Re-derive the exact same sampling as the chunk jobs used
    state   = _load_and_sample(dataset_path, n, seed)
    n_total = len(state["all_fens"])
    if n_total != ref_n_total:
        raise ValueError(
            f"Dataset produced {n_total} positions but partials expected {ref_n_total}. "
            f"Ensure --n and --seed match the chunk jobs."
        )

    # Assemble full evaluation arrays.
    # Use a boolean coverage array to detect both missing and overlapping ranges.
    new_values       = torch.zeros(n_total, dtype=torch.float32)
    new_move_idxs    = torch.full((n_total,), -1, dtype=torch.long)
    all_packed_masks = torch.zeros(n_total, 512, dtype=torch.uint8)
    valid_mask       = torch.zeros(n_total, dtype=torch.bool)
    covered          = np.zeros(n_total, dtype=bool)

    for p in partials:
        s, e = p["start"], p["end"]
        overlap = covered[s:e].any()
        if overlap:
            n_overlap = int(covered[s:e].sum())
            raise ValueError(
                f"Partial chunk {p['chunk_idx']} (positions [{s}, {e})) overlaps "
                f"{n_overlap} positions already covered by an earlier partial. "
                f"Possible duplicate or mismatched --n-chunks."
            )
        covered[s:e] = True
        new_values[s:e]       = p["new_values"]
        new_move_idxs[s:e]    = p["new_move_idxs"]
        all_packed_masks[s:e] = p["all_packed_masks"]
        valid_mask[s:e]       = p["valid_mask"]

    missing_positions = np.where(~covered)[0]
    if len(missing_positions) > 0:
        lo, hi = int(missing_positions[0]), int(missing_positions[-1])
        raise ValueError(
            f"{len(missing_positions)} positions not covered by any partial "
            f"(range [{lo}, {hi}]). Submit missing chunk jobs before merging."
        )

    print(f"\nAll {n_total:,} positions assembled from {len(partials)} partials "
          f"(depth={ref_depth}, n_chunks={ref_n_chunks}).")

    _apply_filter_and_save(
        dataset_state=state,
        new_values=new_values,
        new_move_idxs_sf=new_move_idxs,
        all_packed_masks=all_packed_masks,
        valid_mask=valid_mask,
        out_path=out_path,
        min_decisive=min_decisive,
        max_pieces=max_pieces,
        derive_drawness_from_outcome=derive_drawness_from_outcome,
        drawness_sf_threshold=drawness_sf_threshold,
        drawness_min_ply=drawness_min_ply,
        drawness_target=drawness_target,
        drawness_max_pieces=drawness_max_pieces,
        drawness_game_level=drawness_game_level,
        extra_meta={"source": dataset_path, "stockfish_depth": partials[0]["depth"]},
    )


# ---------------------------------------------------------------------------
# Re-evaluation (single job — original behaviour, preserved for compatibility)
# ---------------------------------------------------------------------------

def reeval(dataset_path: str,
           out_path: str,
           stockfish_path: str = "/usr/games/stockfish",
           depth: int = 10,
           n: int = None,
           seed: int = 42,
           workers: int = 1,
           min_decisive: float = 0.0,
           max_pieces: int = 32,
           derive_drawness_from_outcome: bool = False,
           drawness_sf_threshold: float = 0.15,
           drawness_min_ply: int = 40,
           drawness_target: float = 0.7,
           drawness_max_pieces: int = 32,
           drawness_game_level: bool = False):
    """
    Load dataset, re-evaluate all positions in one job, save new dataset.

    For long runtimes that may exceed the HPC wall time, use the chunked
    workflow instead: --chunk-idx / --n-chunks to evaluate slices, then
    --merge to combine the partial files into the final dataset.

    n:             if set, subsample this many positions from train+val combined.
    workers:       number of parallel Stockfish processes.
    min_decisive:  keep only positions where |tanh(eval)| >= this value.
                   0.0 = keep all. 0.3 ≈ 130cp advantage.
    max_pieces:    keep only positions with at most this many pieces on the board.
                   32 = keep all. 16 = endgame-like. 24 = simplified middlegame.
    """
    print(f"Loading {dataset_path} ...")
    state   = _load_and_sample(dataset_path, n, seed)
    n_total = len(state["all_fens"])
    print(f"  Total positions to evaluate: {n_total:,}")
    print(f"\nStarting {workers} Stockfish worker(s) (depth={depth}) ...")

    new_values, new_move_idxs, all_packed_masks, valid_mask, _ = \
        _run_sf_on_slice(state["all_fens"], stockfish_path, depth, workers)

    _apply_filter_and_save(
        dataset_state=state,
        new_values=new_values,
        new_move_idxs_sf=new_move_idxs,
        all_packed_masks=all_packed_masks,
        valid_mask=valid_mask,
        out_path=out_path,
        min_decisive=min_decisive,
        max_pieces=max_pieces,
        derive_drawness_from_outcome=derive_drawness_from_outcome,
        drawness_sf_threshold=drawness_sf_threshold,
        drawness_min_ply=drawness_min_ply,
        drawness_target=drawness_target,
        drawness_max_pieces=drawness_max_pieces,
        drawness_game_level=drawness_game_level,
        extra_meta={"source": dataset_path, "stockfish_depth": depth},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Re-evaluate dataset positions with Stockfish.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
-----
  Single job (default):
    python3 reeval_stockfish.py --dataset X --out Y [--depth D --workers W ...]

  Chunked (split across multiple HPC jobs, each fits within 24h wall):
    # Run one job per chunk — all must share --n and --seed:
    python3 reeval_stockfish.py --dataset X --n 400000 --seed 42 \\
        --depth 20 --workers 16 --stockfish /path/sf \\
        --chunk-idx 0 --n-chunks 3 --partial-out /scratch/part_0.pt
    # ... repeat for chunk-idx 1, 2

    # Merge when all chunks are done:
    python3 reeval_stockfish.py --dataset X --out Y \\
        --merge /scratch/part_0.pt /scratch/part_1.pt /scratch/part_2.pt \\
        --n 400000 --seed 42 --min-decisive 0.0 --max-pieces 32
"""
    )
    ap.add_argument("--dataset",    required=True,
                    help="Input dataset.pt (required for all modes)")
    ap.add_argument("--out",        default="dataset_sf.pt",
                    help="Output path for final dataset (single-job and merge modes)")
    ap.add_argument("--stockfish",  default="/usr/games/stockfish",
                    help="Stockfish binary path")
    ap.add_argument("--depth",      type=int, default=10,
                    help="Search depth (single-job and chunk modes)")
    ap.add_argument("--n",          type=int, default=200_000,
                    help="Positions to evaluate total (must match across all chunks)")
    ap.add_argument("--seed",       type=int, default=42,
                    help="Random seed for subsampling (must match across all chunks)")
    ap.add_argument("--workers",    type=int, default=1,
                    help="Parallel Stockfish processes (single-job and chunk modes)")
    ap.add_argument("--min-decisive", type=float, default=0.0,
                    help="Keep only |tanh(eval)| >= this. 0.0 = keep all (default).")
    ap.add_argument("--max-pieces",   type=int,   default=32,
                    help="Keep only positions with <= this many pieces (default: 32).")
    ap.add_argument("--derive-drawness-from-outcome", action="store_true",
                    help="After SF re-eval, mark drawn-game positions as soft "
                         "drawness positives when outcome_values==0, |SF eval| "
                         "is small, and ply is late enough. Requires datasets "
                         "parsed with outcome_values and plys.")
    ap.add_argument("--drawness-sf-threshold", type=float, default=0.15,
                    help="SF value threshold for drawn-game drawness positives "
                         "(default: 0.15).")
    ap.add_argument("--drawness-min-ply", type=int, default=40,
                    help="Minimum half-move index for drawn-game drawness positives "
                         "(default: 40).")
    ap.add_argument("--drawness-target", type=float, default=0.7,
                    help="Soft drawness target assigned to drawn-game positives "
                         "(default: 0.7).")
    ap.add_argument("--drawness-max-pieces", type=int, default=32,
                    help="Optional piece-count cap for drawn-game drawness positives "
                         "(default: 32 = no extra cap).")
    ap.add_argument("--drawness-game-level", action="store_true",
                    help="Game-level drawness: mark all positions in a drawn game only "
                         "if the SF eval NEVER exceeded --drawness-sf-threshold for any "
                         "sampled position in that game. Stricter than the default "
                         "per-position check — excludes games where one side was winning "
                         "but blundered back to a draw.")

    # Chunked mode
    ap.add_argument("--chunk-idx",   type=int, default=None,
                    help="Index of this chunk (0-based). Enables chunk mode.")
    ap.add_argument("--n-chunks",    type=int, default=None,
                    help="Total number of chunks. Required with --chunk-idx.")
    ap.add_argument("--partial-out", default=None,
                    help="Output path for this chunk's partial result. "
                         "Required with --chunk-idx.")

    # Merge mode
    ap.add_argument("--merge", nargs="+", default=None, metavar="PARTIAL",
                    help="Merge mode: list of partial .pt files to combine into --out.")

    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT,  _handle_sigterm)

    if args.merge is not None:
        # Merge mode — no SF evaluation, just assemble partials and filter
        if args.out == "dataset_sf.pt" and args.out not in args.merge:
            pass  # user didn't set --out explicitly; warn but proceed
        merge_partials(
            dataset_path=args.dataset,
            partial_paths=args.merge,
            out_path=args.out,
            min_decisive=args.min_decisive,
            max_pieces=args.max_pieces,
            n=args.n,
            seed=args.seed,
            derive_drawness_from_outcome=args.derive_drawness_from_outcome,
            drawness_sf_threshold=args.drawness_sf_threshold,
            drawness_min_ply=args.drawness_min_ply,
            drawness_target=args.drawness_target,
            drawness_max_pieces=args.drawness_max_pieces,
            drawness_game_level=args.drawness_game_level,
        )

    elif args.chunk_idx is not None:
        # Chunk mode — evaluate a slice, save partial
        if args.n_chunks is None:
            ap.error("--n-chunks is required with --chunk-idx")
        if args.partial_out is None:
            ap.error("--partial-out is required with --chunk-idx")
        reeval_chunk(
            dataset_path=args.dataset,
            partial_out=args.partial_out,
            chunk_idx=args.chunk_idx,
            n_chunks=args.n_chunks,
            stockfish_path=args.stockfish,
            depth=args.depth,
            n=args.n,
            seed=args.seed,
            workers=args.workers,
        )

    else:
        # Single-job mode — original behaviour
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
            derive_drawness_from_outcome=args.derive_drawness_from_outcome,
            drawness_sf_threshold=args.drawness_sf_threshold,
            drawness_min_ply=args.drawness_min_ply,
            drawness_target=args.drawness_target,
            drawness_max_pieces=args.drawness_max_pieces,
            drawness_game_level=args.drawness_game_level,
        )


if __name__ == "__main__":
    main()
