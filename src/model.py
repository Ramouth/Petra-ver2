"""
PetraNet — CNN backbone with L2-normalised geometry bottleneck.

Architecture
------------
Input:       (B, 14, 8, 8)  — always STM-relative (see board.py)
ConvBlock:   14 → 64 channels, 3×3
ResBlocks:   64 channels × N_BLOCKS (default 4), each with skip connection
Flatten:     64 × 8 × 8 = 4096
Bottleneck:  Linear(4096 → 128) + L2Norm  ← geometry space (unit hypersphere)

Value head:  Linear(128 → 1) + Tanh  (thin — geometry carries the representation)
Policy head: Linear(128 → 4096)  [logits over STM-relative 64×64 from/to pairs]

The bottleneck projects onto the unit hypersphere via L2 normalisation.
Cosine similarity is therefore the natural metric — consistent with how
probe_geometry.py and test_geometry.py measure the space.

No saturation (unlike Tanh), gradients flow freely to the ResBlocks.
The geometry space is not engineered — it emerges from training.

Value convention: +1 = current side to move wins.
Policy: logits in STM-relative coordinates, masked to legal moves before softmax.
        Always pass flip=(board.turn == chess.BLACK) to move_to_index.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from board import board_to_tensor, move_to_index


class L2Norm(nn.Module):
    """
    Project onto the unit hypersphere via L2 normalisation.

    Makes cosine similarity the natural metric for the geometry space —
    consistent with how probe_geometry.py and test_geometry.py measure it.
    No saturation, no vanishing gradients through the bottleneck.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r))
        return F.relu(r + x)


class PetraNet(nn.Module):
    """
    CNN backbone + 128-dim geometry bottleneck + value and policy heads.
    """

    def __init__(self, n_blocks: int = 4, channels: int = 64, bottleneck_dim: int = 128):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim

        self.input_block = ConvBlock(14, channels)
        self.res_blocks  = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, bottleneck_dim),
            L2Norm(),                  # unit hypersphere — cosine is the natural metric
        )

        self.value_head = nn.Sequential(
            nn.Linear(bottleneck_dim, 1),
            nn.Tanh(),                 # thin head — geometry carries the representation
        )

        self.policy_head = nn.Linear(bottleneck_dim, 64 * 64)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 14, 8, 8)
        Returns: value (B, 1), policy_logits (B, 4096)
        """
        x = self.input_block(x)
        x = self.res_blocks(x)
        g = self.bottleneck(x)
        return self.value_head(g), self.policy_head(g)

    def _geometry_fwd(self, x: torch.Tensor) -> torch.Tensor:
        """
        Geometry forward pass with gradients intact.
        Used by training losses (e.g. concept anchoring) that need to backprop
        through the bottleneck. Not for external use — call geometry() for probing.
        """
        x = self.input_block(x)
        x = self.res_blocks(x)
        return self.bottleneck(x)

    def geometry(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the 128-dim geometry vector for a batch of board tensors.
        No gradient — use for probing and nearest-neighbour queries.
        For training losses, use _geometry_fwd() instead.
        """
        with torch.no_grad():
            return self._geometry_fwd(x)

    @torch.no_grad()
    def policy(self, board: chess.Board, device: torch.device) -> dict:
        """
        Return a probability distribution over legal moves for a single board.
        Returns: {chess.Move: float}

        Move indices are in STM-relative coordinates — flip is applied
        consistently so logits align with the flipped board tensor.
        """
        self.eval()
        t = board_to_tensor(board).unsqueeze(0).to(device)
        _, logits = self.forward(t)
        logits = logits.squeeze(0)

        flip = (board.turn == chess.BLACK)
        mask = torch.full((4096,), float("-inf"), device=device)
        for move in board.legal_moves:
            idx = move_to_index(move, flip=flip)
            mask[idx] = logits[idx]

        probs = torch.softmax(mask, dim=0)
        return {move: probs[move_to_index(move, flip=flip)].item()
                for move in board.legal_moves}

    @torch.no_grad()
    def value(self, board: chess.Board, device: torch.device) -> float:
        """
        Return the scalar value for a single board.
        +1 = current side to move wins.
        """
        self.eval()
        t = board_to_tensor(board).unsqueeze(0).to(device)
        v, _ = self.forward(t)
        return v.item()
