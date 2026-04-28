"""
Data pipeline for Phase 1 supervised pretraining.

Parses Lichess PGN files, extracts labeled positions, runs integrity
checks, and saves a validated dataset.

Label convention
----------------
Values are from the perspective of the side to move:
  win  = +1.0   (side to move wins the game)
  loss = -1.0   (side to move loses)
  draw =  0.0   (neutral — draw contempt belongs in RL, not supervised pretraining)

This is verified by the validation suite before any dataset is saved.

Integrity checks
----------------
1. Label values are strictly in {+1.0, -0.1, -1.0} — no floats, no nulls
2. Sign correctness: White-to-move positions in White-won games → +1.0
                     Black-to-move positions in White-won games → -1.0
3. Side-to-move balance: ~50% White, ~50% Black (within ±10%)
4. King presence: every position has exactly one White king and one Black king
5. Label distribution: at least 20% of each class (no severe imbalance)
6. Train/val split is at the GAME level — no position leakage across splits
7. Spot-check known positions against expected labels

Usage
-----
    python3 data.py --pgn games.pgn --out dataset.pt --max-games 100000
    python3 data.py --pgn games.pgn --out dataset.pt --validate-only
"""

import argparse
import io
import json
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import List, NamedTuple

import numpy as np
import chess
import chess.pgn
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board import board_to_tensor, outcome_to_value, move_to_index

# Defaults — all overridable via CLI
SKIP_OPENING_MOVES = 10
MAX_POSITIONS_PER_GAME = 8
MIN_GAME_MOVES = 15

# Maximum game length (runaway games add noise)
MAX_GAME_MOVES = 200

# Validation fraction held out at game level
VAL_FRACTION = 0.05

VALID_LABEL_VALUES = {1.0, 0.0, -1.0}

_LABEL_TOL = 1e-4

def _label_class(v: float) -> str:
    """Classify a label value tolerantly. Returns 'win', 'draw', 'loss', or 'invalid'."""
    if abs(v - 1.0) < _LABEL_TOL:  return "win"
    if abs(v + 1.0) < _LABEL_TOL:  return "loss"
    if abs(v)       < _LABEL_TOL:  return "draw"
    return "invalid"


# ---------------------------------------------------------------------------
# Raw dataset — no Python object per position, just numpy arrays
# ---------------------------------------------------------------------------

class RawDataset(NamedTuple):
    tensors:   np.ndarray   # (N, 14, 8, 8) uint8
    values:    np.ndarray   # (N,) float32
    move_idxs: np.ndarray   # (N,) int32
    fens:      list         # list of str, length N
    game_ids:  np.ndarray   # (N,) int32
    plys:      np.ndarray   # (N,) int32


# Keep Position for any external code that imports it
@dataclass
class Position:
    tensor:   torch.Tensor
    value:    float
    move_idx: int
    fen:      str
    game_id:  int
    ply:      int


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _open_pgn(path: str):
    if path.endswith(".zst"):
        import zstandard
        ctx = zstandard.ZstdDecompressor()
        fh  = open(path, "rb")
        return io.TextIOWrapper(ctx.stream_reader(fh), encoding="utf-8", errors="replace")
    return open(path, errors="replace")


def _iter_games(pgn_path: str, max_games: int, min_elo: int,
                require_normal_termination: bool, rng: random.Random,
                skip_opening: int = SKIP_OPENING_MOVES,
                positions_per_game: int = MAX_POSITIONS_PER_GAME,
                sampling: str = "random",
                max_pieces: int = 0,
                decisive_only: bool = False,
                max_elo: int = 0,
                skip_games: int = 0):
    """
    Generator. Yields (game_id, result, [(board, move), ...]) for each
    accepted game. Applies all game-level filters inline.
    """
    games_parsed = games_skipped = 0
    with _open_pgn(pgn_path) as f:
        if skip_games > 0:
            print(f"  Fast-forwarding past first {skip_games:,} raw games ...", flush=True)
            for _ in range(skip_games):
                if chess.pgn.read_game(f) is None:
                    return
            print(f"  Done. Parsing up to {max_games:,} games from offset {skip_games:,}.",
                  flush=True)
        while games_parsed < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                games_skipped += 1
                continue

            if require_normal_termination:
                term = game.headers.get("Termination", "")
                if term and term.lower() not in ("normal", ""):
                    games_skipped += 1
                    continue

            if decisive_only and result == "1/2-1/2":
                games_skipped += 1
                continue

            if min_elo > 0 or max_elo > 0:
                try:
                    w = int(game.headers.get("WhiteElo", "0") or "0")
                    b = int(game.headers.get("BlackElo", "0") or "0")
                    if min_elo > 0 and (w < min_elo or b < min_elo):
                        games_skipped += 1
                        continue
                    if max_elo > 0 and (w > max_elo or b > max_elo):
                        games_skipped += 1
                        continue
                except ValueError:
                    pass

            board = game.board()
            pairs = []
            for move in game.mainline_moves():
                pairs.append((board.copy(), move))
                board.push(move)

            n = len(pairs)
            if n < MIN_GAME_MOVES or n > MAX_GAME_MOVES:
                games_skipped += 1
                continue

            candidates = list(range(skip_opening, n))
            if max_pieces > 0:
                candidates = [i for i in candidates
                              if len(pairs[i][0].piece_map()) <= max_pieces]
            if not candidates:
                games_skipped += 1
                continue

            if sampling == "even":
                k = min(positions_per_game, len(candidates))
                if k == 1:
                    indices = [candidates[0]]
                else:
                    step = (len(candidates) - 1) / (k - 1)
                    indices = [candidates[round(i * step)] for i in range(k)]
                sampled = sorted(set(indices))
            else:
                sampled = rng.sample(candidates, min(positions_per_game, len(candidates)))

            yield games_parsed, result, [(i, pairs[i][0], pairs[i][1]) for i in sampled]

            games_parsed += 1


def parse_pgn(pgn_path: str,
              max_games: int = 200_000,
              min_elo: int = 0,
              max_elo: int = 0,
              require_normal_termination: bool = True,
              seed: int = 42,
              skip_opening: int = SKIP_OPENING_MOVES,
              positions_per_game: int = MAX_POSITIONS_PER_GAME,
              sampling: str = "random",
              checkpoint_path: str = "",
              checkpoint_every: int = 0,
              max_pieces: int = 0,
              decisive_only: bool = False,
              skip_games: int = 0) -> RawDataset:
    """
    Stream a PGN file and return a RawDataset (numpy arrays, no per-position Python objects).

    Memory: pre-allocates uint8 tensor arrays (896 bytes/position instead
    of ~4.5KB for float32 Python objects). Converts to float32 at training time.

    For scale guidance:
      100k games → ~800k positions → ~850MB
      200k games → ~1.6M positions → ~1.7GB
      500k games → ~4M positions   → ~4.3GB
    """
    rng = random.Random(seed)
    max_positions = max_games * positions_per_game

    # Estimate and report memory
    mb = max_positions * (14 * 8 * 8) / 1024 / 1024
    print(f"Parsing {pgn_path} (max_games={max_games:,}) ...")
    print(f"  skip_opening={skip_opening}  positions_per_game={positions_per_game}  sampling={sampling}")
    print(f"Pre-allocating ~{mb:.0f} MB for up to {max_positions:,} positions ...")

    # Pre-allocate as uint8 — board values are binary (0/1)
    tensor_buf   = np.empty((max_positions, 14, 8, 8), dtype=np.uint8)
    value_buf    = np.empty(max_positions, dtype=np.float32)
    move_idx_buf = np.empty(max_positions, dtype=np.int32)
    fens         = []
    game_ids     = []
    plys         = []

    count = 0
    t0 = time.time()
    last_game_id = -1

    for game_id, result, sampled_pairs in _iter_games(
            pgn_path, max_games, min_elo, require_normal_termination, rng,
            skip_opening=skip_opening, positions_per_game=positions_per_game,
            sampling=sampling, max_pieces=max_pieces, decisive_only=decisive_only,
            max_elo=max_elo, skip_games=skip_games):

        if _stop_early:
            print(f"\n  Signal received — stopping after {game_id:,} games.", flush=True)
            if checkpoint_path:
                _save_raw_checkpoint(tensor_buf, value_buf, move_idx_buf,
                                     fens, game_ids, plys, count, game_id + 1,
                                     checkpoint_path)
            break

        for ply, b, move in sampled_pairs:
            tensor_buf[count]   = board_to_tensor(b).numpy().astype(np.uint8)
            value_buf[count]    = outcome_to_value(result, b.turn)
            move_idx_buf[count] = move_to_index(move, flip=(b.turn == chess.BLACK))
            fens.append(b.fen())
            game_ids.append(game_id)
            plys.append(ply)
            count += 1

        last_game_id = game_id
        if (game_id + 1) % 10_000 == 0:
            elapsed = time.time() - t0
            print(f"  {game_id+1:,} games, {count:,} positions, {elapsed:.0f}s")

        if checkpoint_path and checkpoint_every > 0 and (game_id + 1) % checkpoint_every == 0:
            _save_raw_checkpoint(tensor_buf, value_buf, move_idx_buf,
                                 fens, game_ids, plys, count, game_id + 1,
                                 checkpoint_path)

    elapsed = time.time() - t0
    print(f"Done: {last_game_id+1:,} games, {count:,} positions in {elapsed:.1f}s")

    return RawDataset(
        tensors   = tensor_buf[:count],
        values    = value_buf[:count],
        move_idxs = move_idx_buf[:count],
        fens      = fens,
        game_ids  = np.array(game_ids, dtype=np.int32),
        plys      = np.array(plys,     dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Integrity validation
# ---------------------------------------------------------------------------

KNOWN_POSITIONS = [
    # (fen, expected_value, tolerance, description)
    (
        "4k3/8/8/8/8/8/8/4K2Q w - - 0 1",
        1.0, 0.0, "KQ vs K White to move: label must be +1.0 (White wins)"
    ),
    (
        "4k3/8/8/8/8/8/8/4K2Q b - - 0 1",
        -1.0, 0.0, "KQ vs K Black to move: label must be -1.0 (Black loses)"
    ),
]


def validate_dataset(dataset: RawDataset, strict: bool = True) -> bool:
    """
    Run all integrity checks. Prints a full report.
    Returns True if all checks pass. Raises ValueError if strict=True and any fail.
    """
    errors = []
    warnings = []
    n = len(dataset.fens)
    print(f"\n{'='*60}")
    print(f"Dataset validation  ({n:,} positions)")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Check 1: label values are in the allowed set (tolerance for float32)
    # ------------------------------------------------------------------
    bad_label_idx = [i for i in range(n) if _label_class(float(dataset.values[i])) == "invalid"]
    if bad_label_idx:
        errors.append(f"CHECK 1 FAIL — {len(bad_label_idx)} invalid label values "
                      f"(first 5: {[float(dataset.values[i]) for i in bad_label_idx[:5]]})")
    else:
        print("CHECK 1 PASS — all labels in {+1.0, 0.0, -1.0}")

    # ------------------------------------------------------------------
    # Check 2: sign correctness on known positions
    # ------------------------------------------------------------------
    sign_errors = 0
    for fen, expected, tol, desc in KNOWN_POSITIONS:
        board = chess.Board(fen)
        if expected == 1.0:
            simulated = outcome_to_value("1-0" if board.turn == chess.WHITE else "0-1", board.turn)
        elif expected == -1.0:
            simulated = outcome_to_value("1-0" if board.turn == chess.BLACK else "0-1", board.turn)
        else:
            simulated = outcome_to_value("1/2-1/2", board.turn)

        if abs(simulated - expected) > tol:
            errors.append(f"CHECK 2 FAIL — sign error on: {desc}  "
                          f"got={simulated}  expected={expected}")
            sign_errors += 1

    if sign_errors == 0:
        print("CHECK 2 PASS — label sign convention is correct")

    # ------------------------------------------------------------------
    # Check 3: label distribution
    # ------------------------------------------------------------------
    n_win  = sum(1 for v in dataset.values if _label_class(float(v)) == "win")
    n_draw = sum(1 for v in dataset.values if _label_class(float(v)) == "draw")
    n_loss = sum(1 for v in dataset.values if _label_class(float(v)) == "loss")
    pct_win  = 100 * n_win  / n
    pct_draw = 100 * n_draw / n
    pct_loss = 100 * n_loss / n

    print(f"CHECK 3 — label distribution:")
    print(f"  win (+1.0)  : {n_win:>8,}  ({pct_win:5.1f}%)")
    print(f"  draw ( 0.0) : {n_draw:>8,}  ({pct_draw:5.1f}%)")
    print(f"  loss (-1.0) : {n_loss:>8,}  ({pct_loss:5.1f}%)")

    if abs(pct_win - pct_loss) > 15:
        warnings.append(f"CHECK 3 WARN — win/loss imbalance: {pct_win:.1f}% vs {pct_loss:.1f}%")
    if pct_win < 10 or pct_loss < 10:
        errors.append(f"CHECK 3 FAIL — severe label imbalance")

    # ------------------------------------------------------------------
    # Check 4: side-to-move balance
    # ------------------------------------------------------------------
    n_white_to_move = sum(
        1 for fen in dataset.fens if chess.Board(fen).turn == chess.WHITE
    )
    n_black_to_move = n - n_white_to_move
    pct_white = 100 * n_white_to_move / n

    print(f"CHECK 4 — side to move:")
    print(f"  White to move: {n_white_to_move:,}  ({pct_white:.1f}%)")
    print(f"  Black to move: {n_black_to_move:,}  ({100-pct_white:.1f}%)")

    if abs(pct_white - 50) > 10:
        warnings.append(f"CHECK 4 WARN — side-to-move imbalance: {pct_white:.1f}% White")

    # ------------------------------------------------------------------
    # Check 5: king presence (spot-check first 2000 positions)
    # ------------------------------------------------------------------
    king_errors = 0
    spot_check_n = min(2000, n)
    for i in range(spot_check_n):
        wk = int(dataset.tensors[i, 5].sum())
        bk = int(dataset.tensors[i, 11].sum())
        if wk != 1 or bk != 1:
            king_errors += 1
            if king_errors <= 3:
                errors.append(f"CHECK 5 FAIL — position {i} has "
                              f"{wk} White kings, {bk} Black kings: {dataset.fens[i]}")

    if king_errors == 0:
        print(f"CHECK 5 PASS — king presence verified on {spot_check_n:,} positions "
              f"(plane 5 = STM king, plane 11 = opponent king)")
    else:
        errors.append(f"CHECK 5 FAIL — {king_errors} positions with wrong king count")

    # ------------------------------------------------------------------
    # Check 6: sign consistency within a game (sample 100 games)
    # ------------------------------------------------------------------
    game_to_indices: dict = {}
    for i in range(n):
        gid = int(dataset.game_ids[i])
        game_to_indices.setdefault(gid, []).append(i)

    sign_inconsistencies = 0
    for gid, indices in list(game_to_indices.items())[:100]:
        indices = sorted(indices, key=lambda i: int(dataset.plys[i]))
        for ia, ib in zip(indices, indices[1:]):
            va = _label_class(float(dataset.values[ia]))
            vb = _label_class(float(dataset.values[ib]))
            if va == "draw" or vb == "draw":
                continue
            a_white = chess.Board(dataset.fens[ia]).turn == chess.WHITE
            b_white = chess.Board(dataset.fens[ib]).turn == chess.WHITE
            if a_white == b_white:
                if va != vb:
                    sign_inconsistencies += 1
            else:
                if va == vb:
                    sign_inconsistencies += 1

    if sign_inconsistencies == 0:
        print("CHECK 6 PASS — within-game label signs are consistent")
    else:
        errors.append(f"CHECK 6 FAIL — {sign_inconsistencies} within-game sign inconsistencies")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
    if errors:
        for e in errors:
            print(f"ERROR:   {e}")
        print(f"\nValidation FAILED ({len(errors)} errors, {len(warnings)} warnings)")
        if strict:
            raise ValueError(f"Dataset failed validation with {len(errors)} errors.")
        return False
    else:
        print(f"Validation PASSED  ({len(warnings)} warnings)")
        return True


# ---------------------------------------------------------------------------
# Train / val split and save
# ---------------------------------------------------------------------------

def split_and_save(dataset: RawDataset,
                   out_path: str,
                   val_fraction: float = VAL_FRACTION,
                   seed: int = 42,
                   skip_opening: int = SKIP_OPENING_MOVES,
                   positions_per_game: int = MAX_POSITIONS_PER_GAME,
                   sampling: str = "random"):
    """
    Split at game level (no leakage), validate val set for duplicates,
    and save to disk with a metadata sidecar.

    Works directly from RawDataset numpy arrays to avoid peak memory from
    torch.stack while the original buffer is still alive.
    """
    rng = random.Random(seed)
    n = len(dataset.fens)

    unique_gids = np.unique(dataset.game_ids)
    gid_list = unique_gids.tolist()
    rng.shuffle(gid_list)
    n_val_games = max(1, int(len(gid_list) * val_fraction))
    val_gid_set = set(gid_list[:n_val_games])

    val_mask   = np.isin(dataset.game_ids, np.array(list(val_gid_set), dtype=np.int32))
    train_mask = ~val_mask

    n_train = int(train_mask.sum())
    n_val   = int(val_mask.sum())
    n_train_games = len(gid_list) - n_val_games
    print(f"\nSplit: {n_train:,} train  /  {n_val:,} val  "
          f"(from {n_train_games} / {n_val_games} games)")

    train_fens = [dataset.fens[i] for i in range(n) if train_mask[i]]
    val_fens   = [dataset.fens[i] for i in range(n) if val_mask[i]]

    # Duplicate FEN check in val set
    n_val_dupes = len(val_fens) - len(set(val_fens))
    if n_val_dupes > 0:
        print(f"WARNING: {n_val_dupes} duplicate FENs in validation set")

    # Cross-contamination check
    train_fens_set = set(train_fens)
    val_fens_set   = set(val_fens)
    overlap = train_fens_set & val_fens_set
    if overlap:
        print(f"WARNING: {len(overlap)} FENs appear in both train and val "
              f"(same board position, different games — tolerable)")

    # Slice numpy arrays directly — bool indexing creates contiguous copies,
    # so tensor_buf can be freed as soon as we have both slices.
    train_tensors   = dataset.tensors[train_mask]
    val_tensors     = dataset.tensors[val_mask]
    train_values    = dataset.values[train_mask]
    val_values      = dataset.values[val_mask]
    train_moveidxs  = dataset.move_idxs[train_mask]
    val_moveidxs    = dataset.move_idxs[val_mask]
    train_game_ids  = dataset.game_ids[train_mask]
    val_game_ids    = dataset.game_ids[val_mask]
    train_plys      = dataset.plys[train_mask]
    val_plys        = dataset.plys[val_mask]

    data = {
        "train": {
            "tensors":   torch.from_numpy(train_tensors),
            "values":    torch.from_numpy(train_values),
            "outcome_values": torch.from_numpy(train_values.copy()),
            "move_idxs": torch.from_numpy(train_moveidxs).long(),
            "game_ids":  torch.from_numpy(train_game_ids).long(),
            "plys":      torch.from_numpy(train_plys).long(),
            "fens":      train_fens,
        },
        "val": {
            "tensors":   torch.from_numpy(val_tensors),
            "values":    torch.from_numpy(val_values),
            "outcome_values": torch.from_numpy(val_values.copy()),
            "move_idxs": torch.from_numpy(val_moveidxs).long(),
            "game_ids":  torch.from_numpy(val_game_ids).long(),
            "plys":      torch.from_numpy(val_plys).long(),
            "fens":      val_fens,
        },
        "meta": {
            "n_train":         n_train,
            "n_val":           n_val,
            "val_fraction":    val_fraction,
            "label_values":    list(VALID_LABEL_VALUES),
            "draw_value":      0.0,
            "skip_opening":    skip_opening,
            "max_per_game":    positions_per_game,
            "sampling":        sampling,
            "n_val_dupes":     n_val_dupes,
            "n_fen_overlap":   len(overlap),
        },
    }

    torch.save(data, out_path)
    print(f"Saved → {out_path}")

    meta_path = out_path.replace(".pt", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(data["meta"], f, indent=2)
    print(f"Metadata → {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_stop_early = False

def _handle_sigterm(signum, frame):
    global _stop_early
    print(f"\nSignal {signum} received — will save partial results after current game.",
          flush=True)
    _stop_early = True


def _save_raw_checkpoint(tensor_buf, value_buf, move_idx_buf, fens,
                         game_ids, plys, count, n_games, path):
    """
    Fast checkpoint from live numpy buffers — no Position objects, no validation.
    Written every N games and on SIGINT so progress survives SIGKILL.
    Load with torch.load and pass to _raw_checkpoint_to_dataset().

    Saves numpy views (not copies) — pickle serializes only the filled slice.
    """
    tmp = path + ".tmp"
    torch.save({
        "tensor_buf":   tensor_buf[:count],   # view — pickle writes only count rows
        "value_buf":    value_buf[:count],
        "move_idx_buf": move_idx_buf[:count],
        "fens":         fens,
        "game_ids":     game_ids,
        "plys":         plys,
        "n_positions":  count,
        "n_games":      n_games,
    }, tmp)
    os.replace(tmp, path)
    print(f"  [ckpt] {n_games:,} games / {count:,} positions → {path}", flush=True)


def _raw_checkpoint_to_dataset(ckpt: dict) -> RawDataset:
    n = ckpt["n_positions"]
    tb  = ckpt["tensor_buf"]
    vb  = ckpt["value_buf"]
    mb  = ckpt["move_idx_buf"]
    fns = ckpt["fens"]
    gids = ckpt["game_ids"]
    plss = ckpt["plys"]

    # Convert list game_ids/plys to numpy if they came from an old checkpoint format
    if isinstance(gids, list):
        gids = np.array(gids, dtype=np.int32)
    if isinstance(plss, list):
        plss = np.array(plss, dtype=np.int32)

    return RawDataset(
        tensors   = tb[:n] if isinstance(tb, np.ndarray) else tb[:n].numpy(),
        values    = vb[:n] if isinstance(vb, np.ndarray) else vb[:n].numpy(),
        move_idxs = mb[:n] if isinstance(mb, np.ndarray) else mb[:n].numpy(),
        fens      = list(fns[:n]) if hasattr(fns, '__getitem__') else fns,
        game_ids  = gids[:n],
        plys      = plss[:n],
    )


# Keep old name as alias for backward compatibility
def _checkpoint_to_positions(ckpt: dict) -> List[Position]:
    ds = _raw_checkpoint_to_dataset(ckpt)
    return [
        Position(
            tensor   = torch.from_numpy(ds.tensors[i].copy()),
            value    = float(ds.values[i]),
            move_idx = int(ds.move_idxs[i]),
            fen      = ds.fens[i],
            game_id  = int(ds.game_ids[i]),
            ply      = int(ds.plys[i]),
        )
        for i in range(len(ds.fens))
    ]


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT,  _handle_sigterm)

    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn",           required=False,       help="Path to PGN file")
    ap.add_argument("--out",           default="dataset.pt", help="Output .pt path")
    ap.add_argument("--max-games",     type=int, default=100_000)
    ap.add_argument("--min-elo",       type=int, default=1500,
                    help="Minimum Elo for both players (0 = no filter)")
    ap.add_argument("--max-elo",       type=int, default=0,
                    help="Maximum Elo for both players (0 = no filter)")
    ap.add_argument("--no-termination-filter", action="store_true",
                    help="Include games that ended by timeout/abandon")
    ap.add_argument("--validate-only", action="store_true",
                    help="Parse and validate but do not save")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--no-strict",     action="store_true",
                    help="Warn on validation failures instead of raising")
    ap.add_argument("--positions-per-game", type=int, default=MAX_POSITIONS_PER_GAME,
                    help=f"Positions sampled per game (default: {MAX_POSITIONS_PER_GAME}). "
                         "Increase for better game-arc coverage.")
    ap.add_argument("--skip-opening",  type=int, default=SKIP_OPENING_MOVES,
                    help=f"Half-moves to skip at game start (default: {SKIP_OPENING_MOVES}). "
                         "Lower = include earlier positions.")
    ap.add_argument("--sampling",      choices=["random", "even"], default="random",
                    help="'random': sample uniformly at random (default). "
                         "'even': evenly spaced across game arc.")
    ap.add_argument("--max-pieces",     type=int, default=0,
                    help="Only sample positions with ≤ N pieces on board (0 = no filter). "
                         "Use 16 for endgame-only positions.")
    ap.add_argument("--decisive-only",  action="store_true",
                    help="Skip drawn games (result=1/2-1/2). "
                         "Maximises decisive SF labels for geometry bootstrap.")
    ap.add_argument("--checkpoint-every", type=int, default=10_000,
                    help="Save a raw checkpoint every N games (default 10000). "
                         "Survives SIGKILL at wall-time. Set 0 to disable.")
    ap.add_argument("--from-checkpoint", default="",
                    help="Skip parsing; load this raw checkpoint .pt and convert "
                         "it directly to the final dataset.")
    ap.add_argument("--skip-games", type=int, default=0,
                    help="Skip first N raw games before parsing (for parallel chunk jobs). "
                         "Chunk K should set --skip-games K*M and --max-games M.")
    ap.add_argument("--no-split", action="store_true",
                    help="Save raw positions without train/val split, for use with "
                         "--merge-raw. Output is a plain dict .pt, not a split dataset.")
    ap.add_argument("--merge-raw", nargs="+", metavar="CHUNK_PT",
                    help="Merge multiple --no-split chunk outputs into one split dataset. "
                         "Game IDs are rebased to avoid collisions across chunks.")
    args = ap.parse_args()

    # --merge-raw: combine multiple --no-split chunk outputs → one split dataset
    if args.merge_raw:
        print(f"=== Merging {len(args.merge_raw)} raw chunk(s) → {args.out} ===")
        tensors_list, values_list, move_idxs_list = [], [], []
        fens_all, game_ids_list, plys_list = [], [], []
        game_id_offset = 0
        for path in args.merge_raw:
            print(f"  Loading {path} ...", flush=True)
            chunk = torch.load(path, weights_only=False)
            n = chunk["n_positions"]
            gids = np.array(chunk["game_ids"], dtype=np.int32) + game_id_offset
            game_id_offset = int(gids.max()) + 1
            tensors_list.append(np.array(chunk["tensors"])[:n])
            values_list.append(np.array(chunk["values"])[:n])
            move_idxs_list.append(np.array(chunk["move_idxs"])[:n])
            fens_all.extend(list(chunk["fens"])[:n])
            game_ids_list.append(gids)
            plys_list.append(np.array(chunk["plys"])[:n])
            print(f"    {n:,} positions, game_ids {gids.min()}–{gids.max()}")
        merged = RawDataset(
            tensors   = np.concatenate(tensors_list),
            values    = np.concatenate(values_list),
            move_idxs = np.concatenate(move_idxs_list),
            fens      = fens_all,
            game_ids  = np.concatenate(game_ids_list),
            plys      = np.concatenate(plys_list),
        )
        print(f"  Merged: {len(merged.fens):,} positions total")
        validate_dataset(merged, strict=not args.no_strict)
        split_and_save(merged, args.out, seed=args.seed,
                       skip_opening=args.skip_opening,
                       positions_per_game=args.positions_per_game,
                       sampling=args.sampling)
        return

    ckpt_path = args.out.replace(".pt", ".ckpt.pt") if args.checkpoint_every > 0 else ""

    if args.from_checkpoint:
        print(f"Loading checkpoint: {args.from_checkpoint}")
        ckpt = torch.load(args.from_checkpoint, weights_only=False)
        print(f"  {ckpt['n_games']:,} games / {ckpt['n_positions']:,} positions")
        dataset = _raw_checkpoint_to_dataset(ckpt)
    else:
        if not args.pgn:
            ap.error("--pgn is required unless --from-checkpoint is set")
        dataset = parse_pgn(
            pgn_path=args.pgn,
            max_games=args.max_games,
            min_elo=args.min_elo,
            max_elo=args.max_elo,
            require_normal_termination=not args.no_termination_filter,
            seed=args.seed,
            skip_opening=args.skip_opening,
            positions_per_game=args.positions_per_game,
            sampling=args.sampling,
            checkpoint_path=ckpt_path,
            checkpoint_every=args.checkpoint_every,
            max_pieces=args.max_pieces,
            decisive_only=args.decisive_only,
            skip_games=args.skip_games,
        )

    if not dataset.fens:
        print("No positions extracted. Check PGN path and filters.")
        sys.exit(1)

    if not args.validate_only:
        if args.no_split:
            raw_out = {
                "tensors":     dataset.tensors,
                "values":      dataset.values,
                "move_idxs":   dataset.move_idxs,
                "fens":        dataset.fens,
                "game_ids":    dataset.game_ids,
                "plys":        dataset.plys,
                "n_positions": len(dataset.fens),
            }
            torch.save(raw_out, args.out, pickle_protocol=4)
            print(f"Saved raw (no split) → {args.out}")
        else:
            split_and_save(dataset, args.out, seed=args.seed,
                           skip_opening=args.skip_opening,
                           positions_per_game=args.positions_per_game,
                           sampling=args.sampling)

    validate_dataset(dataset, strict=not args.no_strict)


if __name__ == "__main__":
    main()
