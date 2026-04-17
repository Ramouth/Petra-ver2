"""
Board representation.

Converts a python-chess Board to a (14, 8, 8) float32 tensor,
always from the perspective of the side to move (STM).

When Black is to move the board is rotated 180° so that:
  - The STM's pieces are always at the bottom (rank 0)
  - Planes 0–5 always encode the STM's pieces
  - Planes 6–11 always encode the opponent's pieces

This means the CNN learns one spatial grammar, not two.
A passed pawn on rank 6, a castled king, a centralised knight —
these look identical to the network regardless of which colour owns them.

  Planes 0–5:   STM pieces      (P N B R Q K)
  Planes 6–11:  Opponent pieces (P N B R Q K)
  Plane 12:     Always 1.0      (STM convention — kept for compat)
  Plane 13:     Castling rights (STM-relative: rank 0 = STM back rank)

Value convention: +1 means the side to move wins.
Move encoding: always in STM-relative coordinates (flip=True when Black to move).
"""

import chess
import torch

# Piece type → plane offset within the 0–5 (STM) or 6–11 (opponent) block.
PIECE_TO_OFFSET = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}


def _flip_square(sq: int) -> int:
    """Flip a square index vertically (rank mirror). File is preserved."""
    return ((7 - (sq >> 3)) << 3) | (sq & 7)


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a python-chess Board to a (14, 8, 8) float32 tensor.

    Always encoded from the STM's perspective.
    When Black is to move, ranks are mirrored so Black's pieces appear
    at rank 0 — identical layout to how White sees its own position.
    """
    t = torch.zeros(14, 8, 8, dtype=torch.float32)
    flip = (board.turn == chess.BLACK)

    for sq, piece in board.piece_map().items():
        rank = sq >> 3
        file = sq & 7
        if flip:
            rank = 7 - rank

        # STM's pieces → planes 0–5, opponent's → planes 6–11
        is_mover = (piece.color == board.turn)
        plane = PIECE_TO_OFFSET[piece.piece_type] + (0 if is_mover else 6)
        t[plane, rank, file] = 1.0

    # Plane 12: always 1.0 — encoding is always from STM's perspective
    t[12] = 1.0

    # Castling: rank 0 = STM back rank, rank 7 = opponent back rank
    stm = board.turn
    opp = not stm
    t[13, 0, 7] = float(board.has_kingside_castling_rights(stm))
    t[13, 0, 0] = float(board.has_queenside_castling_rights(stm))
    t[13, 7, 7] = float(board.has_kingside_castling_rights(opp))
    t[13, 7, 0] = float(board.has_queenside_castling_rights(opp))

    return t


def move_to_index(move: chess.Move, flip: bool = False) -> int:
    """
    Encode a move as from_sq * 64 + to_sq in STM-relative coordinates.

    When flip=True (Black to move), squares are rank-mirrored to match
    the board tensor orientation. Always pass flip=(board.turn == chess.BLACK).
    """
    if flip:
        from_sq = _flip_square(move.from_square)
        to_sq   = _flip_square(move.to_square)
    else:
        from_sq = move.from_square
        to_sq   = move.to_square
    return from_sq * 64 + to_sq


def index_to_squares(idx: int, flip: bool = False):
    """
    Decode a move index to (from_sq, to_sq) in absolute coordinates.

    When flip=True, the stored index is in flipped coordinates —
    squares are un-flipped back to absolute before returning.
    """
    from_sq = idx >> 6
    to_sq   = idx & 63
    if flip:
        from_sq = _flip_square(from_sq)
        to_sq   = _flip_square(to_sq)
    return from_sq, to_sq


def outcome_to_value(result: str, turn: chess.Color) -> float:
    """
    Convert a PGN result string to a value from the perspective of `turn`.

    result: "1-0", "0-1", "1/2-1/2"
    Returns +1 (current side wins), -1 (loses), or 0.0 (draw).
    """
    DRAW_VALUE = 0.0

    if result == "1-0":
        return 1.0 if turn == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if turn == chess.BLACK else -1.0
    return DRAW_VALUE
