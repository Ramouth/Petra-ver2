"""
concepts.py — Chess concept position library for concept anchoring loss.

Concept anchoring: assert that these positions share a structural concept
without assigning a value. The geometry learns what they are worth.
The concepts provide the vocabulary. The model writes the encyclopedia.

Usage (in train.py):
    from concepts import CONCEPTS
    aloss = concept_anchoring_loss(model, CONCEPTS, device, margin=0.3)
    loss = vloss + ploss + 0.05 * aloss

See NOW.md and LATER.md for implementation plan.
See topology_monitor.py (to be built per NOW.md) for concept neighbourhood
analysis — which concepts border the topological holes.
"""

import chess

CONCEPTS = {

    "passed_pawn": [
        # White passed pawn — various ranks and files
        chess.Board("8/8/8/8/P7/8/8/k1K5 w - - 0 1"),
        chess.Board("8/8/8/4P3/8/8/8/k1K5 w - - 0 1"),
        chess.Board("8/8/8/8/8/5P2/8/k1K5 w - - 0 1"),
        chess.Board("8/6P1/8/8/8/8/8/k1K5 w - - 0 1"),
        # Black passed pawn (after board flip: identical structure)
        chess.Board("k1K5/8/8/8/p7/8/8/8 b - - 0 1"),
        chess.Board("k1K5/8/8/4p3/8/8/8/8 b - - 0 1"),
        chess.Board("k1K5/8/p7/8/8/8/8/8 b - - 0 1"),
    ],

    "doubled_rooks": [
        # Rooks doubled on open file
        chess.Board("8/8/8/8/R7/R7/8/k1K5 w - - 0 1"),
        chess.Board("8/8/8/8/4R3/4R3/8/k1K5 w - - 0 1"),
        chess.Board("8/8/3R4/3R4/8/8/8/k1K5 w - - 0 1"),
        # Black doubled rooks
        chess.Board("k1K5/8/r7/r7/8/8/8/8 b - - 0 1"),
        chess.Board("k1K5/4r3/4r3/8/8/8/8/8 b - - 0 1"),
    ],

    "bishop_pair": [
        # Both bishops, opponent has none
        chess.Board("8/8/8/8/B7/2B5/8/k1K5 w - - 0 1"),
        chess.Board("8/8/8/1B6/5B2/8/8/k1K5 w - - 0 1"),
        chess.Board("8/8/3B4/8/1B6/8/8/k1K5 w - - 0 1"),
        chess.Board("k1K5/b7/2b5/8/8/8/8/8 b - - 0 1"),
        chess.Board("k1K5/1b6/5b2/8/8/8/8/8 b - - 0 1"),
    ],

    "rook_on_seventh": [
        # Rook on opponent's second rank
        chess.Board("8/R7/8/8/8/8/8/k1K5 w - - 0 1"),
        chess.Board("8/4R3/8/8/8/8/8/k1K5 w - - 0 1"),
        chess.Board("8/7R/8/8/8/8/8/k1K5 w - - 0 1"),
        chess.Board("k1K5/8/8/8/8/8/r7/8 b - - 0 1"),
        chess.Board("k1K5/8/8/8/8/8/4r3/8 b - - 0 1"),
    ],

    "isolated_pawn": [
        # Isolated pawn — no friendly pawns on adjacent files
        chess.Board("8/8/8/8/4P3/8/8/k1K5 w - - 0 1"),
        chess.Board("8/8/8/2P5/8/8/8/k1K5 w - - 0 1"),
        chess.Board("8/8/8/8/8/6P1/8/k1K5 w - - 0 1"),
        chess.Board("k1K5/8/8/4p3/8/8/8/8 b - - 0 1"),
        chess.Board("k1K5/8/8/2p5/8/8/8/8 b - - 0 1"),
    ],

    "open_file_rook": [
        # Rook on file clear of all pawns
        chess.Board("4R3/8/8/8/8/8/PP2PPPP/k1K5 w - - 0 1"),
        chess.Board("R7/8/8/8/8/8/1PPPPPPP/k1K5 w - - 0 1"),
        chess.Board("7R/8/8/8/8/8/PPPPPPP1/k1K5 w - - 0 1"),
        chess.Board("k1K5/4r3/8/8/8/8/PP2PPPP/8 b - - 0 1"),
        chess.Board("k1K5/r7/8/8/8/8/1PPPPPPP/8 b - - 0 1"),
    ],

    "king_safety_castled": [
        # King behind intact pawn shield, kingside
        chess.Board("8/8/8/8/8/8/5PPP/5RK1 w - - 0 1"),
        chess.Board("8/8/8/8/8/6P1/5PP1/5RK1 w - - 0 1"),
        chess.Board("5rk1/5ppp/8/8/8/8/8/5RK1 b - - 0 1"),
        chess.Board("5rk1/5pp1/6p1/8/8/8/8/5RK1 b - - 0 1"),
    ],

    "outpost_knight": [
        # Knight on central square, not attackable by opponent pawns
        chess.Board("8/8/8/3N4/8/8/8/k1K5 w - - 0 1"),
        chess.Board("8/8/8/8/4N3/8/8/k1K5 w - - 0 1"),
        chess.Board("8/8/4N3/8/8/8/8/k1K5 w - - 0 1"),
        chess.Board("k1K5/8/8/3n4/8/8/8/8 b - - 0 1"),
        chess.Board("k1K5/8/8/8/4n3/8/8/8 b - - 0 1"),
    ],
}


def concept_anchoring_loss(model, concepts: dict, device,
                            margin: float = 0.3,
                            n_pairs: int = 2):
    """
    Concept anchoring loss — nudge geometry toward concept-consistent clusters.

    For each concept, sample pairs of positions and push them together
    on the unit hypersphere. No cross-concept repulsion — the geometry
    organises that naturally through value learning.

    margin: minimum cosine similarity for same-concept pairs (0.3 = gentle nudge)
    n_pairs: pairs sampled per concept per forward pass (keep low)

    Returns scalar loss tensor. Add to main loss with small weight (0.05).
    """
    import torch
    import random
    from board import board_to_tensor

    total_loss = torch.tensor(0.0, device=device)
    n_concepts = 0

    for concept_name, positions in concepts.items():
        if len(positions) < 2:
            continue

        for _ in range(n_pairs):
            i, j = random.sample(range(len(positions)), 2)
            b1, b2 = positions[i], positions[j]

            t1 = board_to_tensor(b1).unsqueeze(0).float().to(device)
            t2 = board_to_tensor(b2).unsqueeze(0).float().to(device)

            g1 = model._geometry_fwd(t1)   # (1, 128) — with gradients for backprop
            g2 = model._geometry_fwd(t2)   # (1, 128)

            cos = (g1 * g2).sum()
            loss = torch.clamp(margin - cos, min=0.0)
            total_loss = total_loss + loss
            n_concepts += 1

    if n_concepts > 0:
        total_loss = total_loss / n_concepts

    return total_loss
