"""
Endgame curriculum position generator for PetraNet.

Generates random legal endgame positions and their antipodal mirror pairs,
labels them by rule (no Stockfish needed for single-piece endgames), and
saves a dataset compatible with train.py.

Label convention — side-to-move relative:
  Stronger side to move → +1.0
  Weaker   side to move → -1.0

Stages
------
  1 = KQ vs K     (queen wins)
  2 = KR vs K     (rook wins)
  3 = KP vs K     (pawn wins — noisy, some draws, useful for pawn geometry)
  4 = KQ vs KR    (queen beats rook)
  5 = KR vs KP    (rook beats pawn)
  6 = KB vs KP    (bishop beats pawn)
  7 = KN vs KP    (knight beats pawn)
  8 = KP vs KP    (more advanced pawn wins — advancement label)

Usage
-----
    # Standalone: build a fixed dataset
    python3 generate_endgame.py --positions 10000 --stages 1 2 --out data/endgame.pt

    # In train.py: per-epoch regeneration (prevents memorisation)
    python3 train.py --endgame-positions 20000 --endgame-stages 1 2 --out models/endgame/
"""

import argparse
import os
import random
import sys

import chess
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board import board_to_tensor, move_to_index


# ---------------------------------------------------------------------------
# Position generators — one per stage
# ---------------------------------------------------------------------------

def random_kqk_position(white_has_queen: bool = True) -> chess.Board:
    """KQ vs K. white_has_queen=False gives the antipodal mirror."""
    while True:
        squares = random.sample(range(64), 3)
        wk_sq, piece_sq, bk_sq = squares

        board = chess.Board(fen=None)
        board.clear()
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        if white_has_queen:
            board.set_piece_at(piece_sq, chess.Piece(chess.QUEEN, chess.WHITE))
        else:
            board.set_piece_at(piece_sq, chess.Piece(chess.QUEEN, chess.BLACK))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE

        if not board.is_valid() or board.is_game_over():
            continue
        return board


def random_krk_position(white_has_rook: bool = True) -> chess.Board:
    """KR vs K. white_has_rook=False gives the antipodal mirror."""
    while True:
        squares = random.sample(range(64), 3)
        wk_sq, piece_sq, bk_sq = squares

        board = chess.Board(fen=None)
        board.clear()
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        if white_has_rook:
            board.set_piece_at(piece_sq, chess.Piece(chess.ROOK, chess.WHITE))
        else:
            board.set_piece_at(piece_sq, chess.Piece(chess.ROOK, chess.BLACK))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE

        if not board.is_valid() or board.is_game_over():
            continue
        return board


def random_kpk_position(white_has_pawn: bool = True) -> chess.Board:
    """KP vs K. Pawn on ranks 2–7 (never back rank or 8th). Labels are noisy."""
    pawn_squares = [sq for sq in range(64) if 1 <= chess.square_rank(sq) <= 6]
    while True:
        pawn_sq = random.choice(pawn_squares)
        remaining = [sq for sq in range(64) if sq != pawn_sq]
        wk_sq, bk_sq = random.sample(remaining, 2)

        board = chess.Board(fen=None)
        board.clear()
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        if white_has_pawn:
            board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.WHITE))
        else:
            board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE

        if not board.is_valid() or board.is_game_over():
            continue
        return board


def random_4piece_position(white_piece: int, black_piece: int,
                            white_piece_on_ranks=None,
                            black_piece_on_ranks=None) -> chess.Board:
    """
    4-piece position: WK + white_piece vs BK + black_piece.
    Caller decides which side is stronger and assigns label accordingly.
    """
    def valid_squares(ranks):
        if ranks is not None:
            return [sq for sq in range(64) if chess.square_rank(sq) in ranks]
        return list(range(64))

    wp_squares = valid_squares(white_piece_on_ranks)
    bp_squares = valid_squares(black_piece_on_ranks)

    while True:
        wp_sq = random.choice(wp_squares)
        bp_sq = random.choice(bp_squares)
        if wp_sq == bp_sq:
            continue
        remaining = [sq for sq in range(64) if sq not in (wp_sq, bp_sq)]
        wk_sq, bk_sq = random.sample(remaining, 2)

        board = chess.Board(fen=None)
        board.clear()
        board.set_piece_at(wk_sq, chess.Piece(chess.KING,  chess.WHITE))
        board.set_piece_at(wp_sq, chess.Piece(white_piece, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING,  chess.BLACK))
        board.set_piece_at(bp_sq, chess.Piece(black_piece, chess.BLACK))
        board.turn = chess.WHITE

        if not board.is_valid() or board.is_game_over():
            continue
        return board


def random_krkr_position() -> chess.Board:
    """KR vs KR — theoretically drawn; label 0.0."""
    while True:
        squares = random.sample(range(64), 4)
        wk_sq, wr_sq, bk_sq, br_sq = squares

        board = chess.Board(fen=None)
        board.clear()
        board.set_piece_at(wk_sq, chess.Piece(chess.KING,  chess.WHITE))
        board.set_piece_at(wr_sq, chess.Piece(chess.ROOK,  chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING,  chess.BLACK))
        board.set_piece_at(br_sq, chess.Piece(chess.ROOK,  chess.BLACK))
        board.turn = chess.WHITE

        if not board.is_valid() or board.is_game_over():
            continue
        return board


def random_knnk_position() -> chess.Board:
    """
    KNN vs K — two knights cannot force checkmate; theoretically drawn; label 0.0.
    Generates both orientations (white has knights OR black has knights) via the
    same function — caller flips color externally if needed, or we just generate
    both white-knight and black-knight variants at equal frequency.
    """
    while True:
        squares = random.sample(range(64), 4)
        wk_sq, wn1_sq, wn2_sq, bk_sq = squares

        board = chess.Board(fen=None)
        board.clear()
        board.set_piece_at(wk_sq,  chess.Piece(chess.KING,   chess.WHITE))
        board.set_piece_at(wn1_sq, chess.Piece(chess.KNIGHT, chess.WHITE))
        board.set_piece_at(wn2_sq, chess.Piece(chess.KNIGHT, chess.WHITE))
        board.set_piece_at(bk_sq,  chess.Piece(chess.KING,   chess.BLACK))
        board.turn = chess.WHITE

        if not board.is_valid() or board.is_game_over():
            continue
        return board


def random_knnk_mirror_position() -> chess.Board:
    """KNN vs K with black holding the knights."""
    while True:
        squares = random.sample(range(64), 4)
        wk_sq, bk_sq, bn1_sq, bn2_sq = squares

        board = chess.Board(fen=None)
        board.clear()
        board.set_piece_at(wk_sq,  chess.Piece(chess.KING,   chess.WHITE))
        board.set_piece_at(bk_sq,  chess.Piece(chess.KING,   chess.BLACK))
        board.set_piece_at(bn1_sq, chess.Piece(chess.KNIGHT, chess.BLACK))
        board.set_piece_at(bn2_sq, chess.Piece(chess.KNIGHT, chess.BLACK))
        board.turn = chess.WHITE

        if not board.is_valid() or board.is_game_over():
            continue
        return board


def random_kbkb_position() -> chess.Board:
    """KB vs KB — equal minor pieces; drawn; label 0.0."""
    while True:
        squares = random.sample(range(64), 4)
        wk_sq, wb_sq, bk_sq, bb_sq = squares

        board = chess.Board(fen=None)
        board.clear()
        board.set_piece_at(wk_sq, chess.Piece(chess.KING,   chess.WHITE))
        board.set_piece_at(wb_sq, chess.Piece(chess.BISHOP, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING,   chess.BLACK))
        board.set_piece_at(bb_sq, chess.Piece(chess.BISHOP, chess.BLACK))
        board.turn = chess.WHITE

        if not board.is_valid() or board.is_game_over():
            continue
        return board


def random_kp_kp_position(white_more_advanced: bool = True) -> chess.Board:
    """
    KP vs KP — both sides have a pawn, more-advanced pawn wins.
    Labels by pawn rank; skips equal-advancement (ambiguous).
    """
    pawn_ranks = list(range(1, 7))
    while True:
        wp_sq = random.choice([sq for sq in range(64) if chess.square_rank(sq) in pawn_ranks])
        bp_sq = random.choice([sq for sq in range(64) if chess.square_rank(sq) in pawn_ranks])
        if wp_sq == bp_sq:
            continue

        white_adv = chess.square_rank(wp_sq)       # higher = closer to promotion
        black_adv = 7 - chess.square_rank(bp_sq)   # lower rank = closer for black

        if white_adv == black_adv:
            continue
        if white_more_advanced and white_adv <= black_adv:
            continue
        if not white_more_advanced and black_adv <= white_adv:
            continue

        remaining = [sq for sq in range(64) if sq not in (wp_sq, bp_sq)]
        wk_sq, bk_sq = random.sample(remaining, 2)

        board = chess.Board(fen=None)
        board.clear()
        board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(wp_sq, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(bp_sq, chess.Piece(chess.PAWN, chess.BLACK))
        board.turn = chess.WHITE

        if not board.is_valid() or board.is_game_over():
            continue
        return board


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

_PAWN_RANKS = list(range(1, 7))

_STAGE_GENERATORS = {
    # (pos_fn, mirror_fn, label_type)
    # 'decisive': white wins in pos_fn, black wins in mirror_fn → labels ±1.0
    # 'draw':     both fns return drawn positions → label always 0.0
    1: (lambda: random_kqk_position(white_has_queen=True),
        lambda: random_kqk_position(white_has_queen=False), 'decisive'),
    2: (lambda: random_krk_position(white_has_rook=True),
        lambda: random_krk_position(white_has_rook=False),  'decisive'),
    3: (lambda: random_kpk_position(white_has_pawn=True),
        lambda: random_kpk_position(white_has_pawn=False),  'decisive'),
    4: (lambda: random_4piece_position(chess.QUEEN, chess.ROOK),
        lambda: random_4piece_position(chess.ROOK,  chess.QUEEN), 'decisive'),
    5: (lambda: random_4piece_position(chess.ROOK, chess.PAWN,
                                        black_piece_on_ranks=_PAWN_RANKS),
        lambda: random_4piece_position(chess.PAWN, chess.ROOK,
                                        white_piece_on_ranks=_PAWN_RANKS), 'decisive'),
    6: (lambda: random_4piece_position(chess.BISHOP, chess.PAWN,
                                        black_piece_on_ranks=_PAWN_RANKS),
        lambda: random_4piece_position(chess.PAWN, chess.BISHOP,
                                        white_piece_on_ranks=_PAWN_RANKS), 'decisive'),
    7: (lambda: random_4piece_position(chess.KNIGHT, chess.PAWN,
                                        black_piece_on_ranks=_PAWN_RANKS),
        lambda: random_4piece_position(chess.PAWN, chess.KNIGHT,
                                        white_piece_on_ranks=_PAWN_RANKS), 'decisive'),
    8: (lambda: random_kp_kp_position(white_more_advanced=True),
        lambda: random_kp_kp_position(white_more_advanced=False), 'decisive'),
    # --- Drawn stages ---
    # label_type='draw' → both fns generate drawn positions; label always 0.0.
    # No color-swap mirror needed (material is symmetric or unwinnable by theory).
    9:  (random_krkr_position,      random_krkr_position,      'draw'),
    10: (random_knnk_position,      random_knnk_mirror_position, 'draw'),
    11: (random_kbkb_position,      random_kbkb_position,      'draw'),
}

STAGE_NAMES = {
    1: "KQ vs K",  2: "KR vs K",   3: "KP vs K",
    4: "KQ vs KR", 5: "KR vs KP",  6: "KB vs KP",
    7: "KN vs KP", 8: "KP vs KP",
    9: "KR vs KR (draw)", 10: "KNN vs K (draw)", 11: "KB vs KB (draw)",
}


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------

def _color_swap(board: chess.Board) -> chess.Board:
    """
    Return the color-swapped antipodal partner of board.

    Same piece types at the same squares, but all piece colors inverted.
    Turn is preserved. This is the true antipodal partner under STM encoding:
    - Original (white has winning piece, white to move) → label +1.0
    - Swapped  (black has winning piece, white to move) → label -1.0

    The spatial layout is identical, so the CNN sees the same board structure
    but with pieces belonging to the opposite sides, forcing the geometry to
    develop win/loss separation for this exact configuration.

    Returns None if the swapped board is invalid (king in check, etc.).
    """
    swapped = chess.Board(fen=None)
    swapped.clear()
    for sq, piece in board.piece_map().items():
        swapped.set_piece_at(sq, chess.Piece(piece.piece_type, not piece.color))
    swapped.turn = board.turn
    if not swapped.is_valid() or swapped.is_game_over():
        return None
    return swapped


def generate_positions(n: int, include_mirrors: bool = True, stages=None):
    """
    Generate n endgame positions, mixed across one or more stages.

    Each position is generated in two turn variants:
      - White to move  (+1.0 if white wins, -1.0 if black wins)
      - Black to move  (label flipped vs W2M)

    If include_mirrors=True (default), each position is also paired with its
    color-swapped antipodal mirror, giving up to 4 variants per sampled position.
    The mirror is the SAME position with piece colors inverted — a true antipodal
    partner that forces win/loss geometric separation.

    Returns list of (board, value) tuples, shuffled.
    """
    if stages is None:
        stages = [1]
    if isinstance(stages, int):
        stages = [stages]

    for s in stages:
        if s not in _STAGE_GENERATORS:
            raise ValueError(f"Unknown endgame stage: {s}. Valid: {sorted(_STAGE_GENERATORS)}")

    n_per_stage = [n // len(stages)] * len(stages)
    n_per_stage[-1] += n - sum(n_per_stage)

    all_positions = []
    for stage, n_stage in zip(stages, n_per_stage):
        pos_fn, mirror_fn, label_type = _STAGE_GENERATORS[stage]
        generated = 0
        seen_fens = set()

        while generated < n_stage:
            board = pos_fn()
            fen = board.board_fen()
            if fen in seen_fens:
                continue
            seen_fens.add(fen)

            if label_type == 'draw':
                # Drawn endgame: label 0.0 regardless of turn or color.
                all_positions.append((board, 0.0))
                board_btm = board.copy()
                board_btm.turn = chess.BLACK
                if board_btm.is_valid() and not board_btm.is_game_over():
                    all_positions.append((board_btm, 0.0))

                if include_mirrors:
                    # For drawn stages, mirror_fn generates a fresh position of the
                    # same material type (no color-swap needed — material is symmetric).
                    mirror = mirror_fn()
                    m_fen = mirror.board_fen()
                    if m_fen not in seen_fens:
                        seen_fens.add(m_fen)
                        all_positions.append((mirror, 0.0))
                        mirror_btm = mirror.copy()
                        mirror_btm.turn = chess.BLACK
                        if mirror_btm.is_valid() and not mirror_btm.is_game_over():
                            all_positions.append((mirror_btm, 0.0))
            else:
                # Decisive endgame: white wins in pos_fn, black wins in mirror_fn.
                # White to move: white wins → +1.0
                all_positions.append((board, +1.0))

                # Same position, black to move: white still has piece but black to move → -1.0
                board_btm = board.copy()
                board_btm.turn = chess.BLACK
                if board_btm.is_valid() and not board_btm.is_game_over():
                    all_positions.append((board_btm, -1.0))

                if include_mirrors:
                    # Color-swapped antipodal: same squares, piece colors inverted.
                    # White to move but white now has the bare king → white loses → -1.0
                    mirror = _color_swap(board)
                    if mirror is not None:
                        all_positions.append((mirror, -1.0))
                        # Black to move with the winning piece → STM (black) wins → +1.0
                        mirror_btm = mirror.copy()
                        mirror_btm.turn = chess.BLACK
                        if mirror_btm.is_valid() and not mirror_btm.is_game_over():
                            all_positions.append((mirror_btm, +1.0))

            generated += 1

    random.shuffle(all_positions)
    return all_positions


# ---------------------------------------------------------------------------
# Optional Stockfish verification
# ---------------------------------------------------------------------------

def label_with_stockfish(positions, stockfish_path: str, depth: int = 5):
    """
    Replace rule-based labels with Stockfish evaluations.
    Useful for verification of stages 3–8 where rule labels are approximate.
    Returns list of (board, value) with SF-derived STM-relative values.
    """
    import math
    import chess.engine

    print(f"  Verifying/re-labelling with Stockfish depth {depth} ...")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    result = []
    for i, (board, _) in enumerate(positions):
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()
        if score.is_mate():
            m = score.mate()
            val = 1.0 if m > 0 else -1.0
        else:
            cp = score.score(mate_score=10000)
            val = math.tanh(cp / 400.0)
        # Convert to STM-relative
        if board.turn == chess.BLACK:
            val = -val
        result.append((board, val))
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(positions)}")
    engine.quit()
    return result


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(positions, val_frac: float = 0.1,
                   store_visit_dists: bool = True):
    """
    Convert list of (board, value) to a train.py-compatible dataset dict.

    store_visit_dists=False: skip the 4096-float visit_dists tensor entirely.
    Use this when building a raw intermediate file for SF re-evaluation —
    reeval_stockfish.py only reads FENs and tensors; visit_dists are wasted
    memory (4096 floats × N positions can be tens of GB for large datasets).

    Policy target: uniform over legal moves. Policy loss on endgame positions
    without MCTS is not meaningful — use --policy-weight 0 in train.py to
    train value-only on the endgame curriculum.

    Important: move_to_index is called with flip=(board.turn == chess.BLACK)
    so move indices are always in STM-relative coordinates, matching the
    board tensor representation.
    """
    random.shuffle(positions)
    n_val = max(1, int(len(positions) * val_frac))
    splits = {
        "train": positions[n_val:],
        "val":   positions[:n_val],
    }

    data = {}
    for split_name, split in splits.items():
        tensors   = []
        values    = []
        move_idxs = []
        vds       = [] if store_visit_dists else None
        fens      = []

        for board, value in split:
            tensors.append(board_to_tensor(board))
            values.append(value)
            fens.append(board.fen())

            flip  = (board.turn == chess.BLACK)
            legal = list(board.legal_moves)
            if legal:
                move_idxs.append(move_to_index(legal[0], flip=flip))
            else:
                move_idxs.append(0)

            if store_visit_dists:
                vd = torch.zeros(4096, dtype=torch.float32)
                if legal:
                    w = 1.0 / len(legal)
                    for m in legal:
                        vd[move_to_index(m, flip=flip)] = w
                vds.append(vd)

        split_data = {
            "tensors":   torch.stack(tensors).to(torch.uint8),
            "values":    torch.tensor(values, dtype=torch.float32),
            "move_idxs": torch.tensor(move_idxs, dtype=torch.int64),
            "fens":      fens,
        }
        if store_visit_dists:
            split_data["visit_dists"] = torch.stack(vds)
        data[split_name] = split_data

    n_train    = len(data["train"]["tensors"])
    n_val_act  = len(data["val"]["tensors"])
    pos_values = data["train"]["values"]
    wins       = (pos_values >  0.5).sum().item()
    losses     = (pos_values < -0.5).sum().item()

    data["meta"] = {
        "source":     "endgame",
        "n_train":    n_train,
        "n_val":      n_val_act,
        "label_mean": float(pos_values.mean()),
        "label_std":  float(pos_values.std()),
        "pct_win":    wins   / n_train,
        "pct_loss":   losses / n_train,
    }

    print(f"  train: {n_train:,}  val: {n_val_act:,}")
    print(f"  labels — win: {wins/n_train*100:.1f}%  "
          f"loss: {losses/n_train*100:.1f}%  "
          f"mean: {float(pos_values.mean()):+.3f}")

    return data


# ---------------------------------------------------------------------------
# CLI — standalone dataset builder
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate endgame curriculum dataset for PetraNet training."
    )
    ap.add_argument("--positions",  type=int, default=10000,
                    help="Number of base positions per stage (mirrors included automatically)")
    ap.add_argument("--stages",     type=int, nargs="+", default=[1],
                    help="Stages to mix: 1=KQK 2=KRK 3=KPK 4=KQvKR 5=KRvKP "
                         "6=KBvKP 7=KNvKP 8=KPvKP "
                         "9=KRvKR(draw) 10=KNNvK(draw) 11=KBvKB(draw) (default: 1)")
    ap.add_argument("--out",        required=True,
                    help="Output .pt file path")
    ap.add_argument("--no-mirrors", action="store_true",
                    help="Skip antipodal mirror positions (not recommended)")
    ap.add_argument("--no-visit-dists", action="store_true",
                    help="Skip storing visit_dists (4096-float policy targets). "
                         "Use when building a raw intermediate file for SF re-evaluation "
                         "— saves ~16GB for 1M positions.")
    ap.add_argument("--stockfish",  default=None,
                    help="Stockfish binary path for label verification (optional)")
    ap.add_argument("--depth",      type=int, default=5,
                    help="Stockfish depth for verification (default: 5)")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    include_mirrors = not args.no_mirrors
    stage_label = "+".join(STAGE_NAMES.get(s, f"stage{s}") for s in args.stages)

    print(f"Generating {args.positions:,} × {stage_label}"
          + (" + mirrors" if include_mirrors else "") + " ...")

    positions = generate_positions(args.positions,
                                   include_mirrors=include_mirrors,
                                   stages=args.stages)
    print(f"  Total positions: {len(positions):,}")

    if args.stockfish:
        positions = label_with_stockfish(positions, args.stockfish, args.depth)

    print("\nBuilding dataset ...")
    data = build_dataset(positions, store_visit_dists=not args.no_visit_dists)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(data, args.out)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
