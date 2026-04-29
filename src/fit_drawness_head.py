"""
fit_drawness_head.py — fit the drawness head via sklearn logistic regression
on frozen geometry embeddings, then write the optimal weights into the model.

Adam-based training stalls: balanced positions converge to 0 but structural
draws plateau at 0.5 (DrawL stuck at 0.585 after epoch 6). Root cause: Adam
found a basin where the non-draw gradient dominates; the structural draw class
never gets pushed above the sigmoid midpoint.

sklearn's L-BFGS finds the global BCE minimum in one shot. The probe already
confirmed 99% linear separability (Cohen's d=3.769) on this geometry — the
hyperplane exists, we just need to find it correctly.

Usage (HPC):
    python3 src/fit_drawness_head.py \
        --model  models/phase15_mid_no_endgame/best.pt \
        --out    models/drawness_bootstrap/best.pt \
        --n      50000
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import PetraNet
from generate_endgame import generate_positions
from board import board_to_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True,  help="Path to backbone .pt")
    ap.add_argument("--out",    required=True,  help="Output model path")
    ap.add_argument("--n",      type=int, default=50000,
                    help="Positions per class (draws and decisive each)")
    ap.add_argument("--C",      type=float, default=1.0,
                    help="sklearn LR regularisation strength")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model with frozen backbone
    model = PetraNet().to(device)
    sd = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Loaded {args.model}")

    # Generate balanced sample: args.n structural draws + args.n decisive positions
    draw_stages    = [9, 10, 11]          # KR vs KR, KNN vs K, KB vs KB
    decisive_stages = [1, 2, 4, 5]       # KQ vs K, KR vs K, KQ vs KR, KR vs KP

    print(f"Generating {args.n} structural draw positions (stages {draw_stages})...")
    draw_positions = generate_positions(args.n, include_mirrors=False, stages=draw_stages)

    print(f"Generating {args.n} decisive positions (stages {decisive_stages})...")
    decisive_positions = generate_positions(args.n, include_mirrors=False, stages=decisive_stages)

    # Convert boards to geometry vectors via frozen backbone
    def boards_to_embeddings(positions):
        tensors = torch.stack([
            board_to_tensor(board) for board, _, _ in positions
        ]).float().to(device)
        batch_size = 1024
        gs = []
        with torch.no_grad():
            for i in range(0, len(tensors), batch_size):
                g = model._geometry_fwd(tensors[i:i+batch_size])
                gs.append(g.cpu().numpy())
        return np.vstack(gs).astype(np.float32)

    print("Computing draw embeddings...")
    X_draw = boards_to_embeddings(draw_positions)
    print("Computing decisive embeddings...")
    X_dec  = boards_to_embeddings(decisive_positions)

    X = np.vstack([X_draw, X_dec])
    y = np.array([1] * len(X_draw) + [0] * len(X_dec), dtype=np.int32)
    print(f"Dataset: {len(X_draw)} draws + {len(X_dec)} decisive = {len(X)} total")

    # Fit logistic regression to global optimum
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    print(f"Fitting sklearn LR (C={args.C}, solver=lbfgs)...")
    clf = LogisticRegression(max_iter=2000, C=args.C, solver="lbfgs")
    cv  = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

    clf.fit(X, y)

    # Transfer weights directly into drawness_head (Linear + Sigmoid)
    # sklearn: P(draw|g) = sigmoid(coef_ @ g + intercept_)
    # drawness_head: sigmoid(weight @ g + bias)  — identical form
    w = torch.tensor(clf.coef_,      dtype=torch.float32)   # (1, 128)
    b = torch.tensor(clf.intercept_, dtype=torch.float32)   # (1,)

    with torch.no_grad():
        model.drawness_head[0].weight.copy_(w)
        model.drawness_head[0].bias.copy_(b)

    # Sanity check
    import chess
    positions_probe = [
        ("KR vs KR (structural draw)",       "8/3k4/8/r7/8/8/3K4/7R w - - 0 1",         ">0.7"),
        ("KQ vs K (decisive)",               "8/8/8/8/8/8/1K6/Q6k w - - 0 1",            "<0.3"),
        ("Sicilian (balanced, sharp)",       "rnbqkbnr/pp1ppppp/8/2p5/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", "<0.3"),
    ]
    print("\nSanity check:")
    for name, fen, gate in positions_probe:
        board = chess.Board(fen)
        d = model.drawness(board, device)
        mark = "✓" if (gate.startswith(">") and d > float(gate[1:])) or \
                      (gate.startswith("<") and d < float(gate[1:])) else "✗"
        print(f"  {mark} {name:45s}  drawness={d:.3f}  (gate: {gate})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
