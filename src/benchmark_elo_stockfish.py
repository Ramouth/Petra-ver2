"""
benchmark_elo_stockfish.py — calibrate a Petra model's playing strength
against Stockfish at multiple Skill Level settings.

Stockfish's UCI `Skill Level` option (0-20) approximates a graded
playing strength. By matching our model against several skill levels
and finding the level where we score ~50%, we get a calibrated ELO
estimate suitable for the paper.

Reference Skill-Level → ELO table (Stockfish docs / community estimates):
    Skill 0  ≈ 1320
    Skill 5  ≈ 1500
    Skill 10 ≈ 1750
    Skill 15 ≈ 2200
    Skill 20 = full strength (~3500)

Usage
-----
    python3 src/benchmark_elo_stockfish.py \\
        --model       /path/to/petra/best.pt \\
        --stockfish   /path/to/stockfish \\
        --skill-levels 0 5 10 \\
        --games       100 \\
        --n-sim       100 \\
        --out         results.json
"""

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path

import chess
import chess.engine
import torch

# Make src/ importable when run via `python3 src/benchmark_elo_stockfish.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Agent, MCTSAgent, play_game
from model import PetraNet


# Linear interpolation table — Stockfish skill levels to approximate ELO.
# These are community estimates, not official; documented in the paper as
# such. The actual calibration comes from the matchups, not this table.
SKILL_TO_ELO = {
    0: 1320, 1: 1370, 2: 1410, 3: 1440, 4: 1470,
    5: 1500, 6: 1540, 7: 1580, 8: 1630, 9: 1690,
    10: 1750, 11: 1820, 12: 1900, 13: 1980, 14: 2080,
    15: 2200, 16: 2350, 17: 2500, 18: 2700, 19: 3000, 20: 3500,
}


class StockfishAgent(Agent):
    """Stockfish UCI engine constrained by Skill Level + small think-time.

    Time budget kept small so Stockfish-skill is the binding constraint
    on strength, not depth. We want the published Skill-Level ELO bands
    to apply, which assume modest think time.
    """

    def __init__(self, stockfish_path: str, skill_level: int,
                 think_time_ms: int = 100):
        if not 0 <= skill_level <= 20:
            raise ValueError(f"skill_level must be 0..20, got {skill_level}")
        self._path = stockfish_path
        self._skill = skill_level
        self._think_time_ms = think_time_ms
        self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self._engine.configure({"Skill Level": skill_level})

    def select_move(self, board: chess.Board) -> chess.Move:
        result = self._engine.play(
            board,
            chess.engine.Limit(time=self._think_time_ms / 1000.0),
        )
        return result.move

    def close(self):
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def __del__(self):
        self.close()

    @property
    def name(self):
        return f"Stockfish(skill={self._skill})"

    @property
    def cfg(self):
        return {"type": "stockfish", "skill": self._skill,
                "think_time_ms": self._think_time_ms,
                "path": self._path}


def play_match(petra_agent: MCTSAgent, sf_path: str, skill: int,
               n_games: int, think_time_ms: int) -> dict:
    """Play n_games between petra and stockfish at given skill.

    Petra plays half the games as White, half as Black. Game results
    are aggregated from petra's perspective.
    """
    sf = StockfishAgent(sf_path, skill, think_time_ms=think_time_ms)

    wins = draws = losses = 0
    for game_idx in range(n_games):
        if game_idx % 2 == 0:
            white, black = petra_agent, sf
            petra_is_white = True
        else:
            white, black = sf, petra_agent
            petra_is_white = False

        result, _ = play_game(white, black, max_moves=300)

        if result == "1-0":
            if petra_is_white: wins += 1
            else:               losses += 1
        elif result == "0-1":
            if petra_is_white: losses += 1
            else:               wins += 1
        else:
            draws += 1

        if (game_idx + 1) % 20 == 0 or (game_idx + 1) == n_games:
            score = wins + 0.5 * draws
            wr = score / (game_idx + 1)
            print(f"  [skill={skill:>2}]  [{game_idx+1:>4}/{n_games}]  "
                  f"W={wins} D={draws} L={losses}  wr={wr:.3f}", flush=True)

    sf.close()

    score = wins + 0.5 * draws
    wr = score / n_games
    return {
        "skill": skill,
        "stockfish_elo_est": SKILL_TO_ELO[skill],
        "games": n_games,
        "wins": wins, "draws": draws, "losses": losses,
        "score": score, "wr": wr,
    }


def estimate_elo(results: list) -> dict:
    """Given match results at multiple skill levels, estimate Petra's ELO.

    Standard Bayesian-ish approach: each match gives an ELO delta vs the
    opponent's published ELO (delta = -400 * log10(1/wr - 1)). Aggregate
    by inverse-variance weighting (variance ~ 1/n for each match).
    """
    deltas = []
    weights = []
    for r in results:
        wr = r["wr"]
        if wr <= 0.0 or wr >= 1.0:
            # Saturated; treat as ±400 ELO with low confidence
            delta = 400.0 if wr >= 1.0 else -400.0
            w = max(1, r["games"]) * 0.25
        else:
            delta = -400.0 * math.log10((1.0 / wr) - 1.0)
            # Variance ~ wr*(1-wr)/n — convert to inverse-variance weight
            var_wr = max(1e-6, wr * (1.0 - wr) / max(1, r["games"]))
            # Weight in log-odds space (rough): scale by n*wr*(1-wr)
            w = r["games"] * wr * (1.0 - wr)

        deltas.append(r["stockfish_elo_est"] + delta)
        weights.append(w)

    total_w = sum(weights)
    if total_w == 0:
        return {"elo_est": None, "elo_low": None, "elo_high": None,
                "method": "no usable matches"}

    weighted_mean = sum(d * w for d, w in zip(deltas, weights)) / total_w
    # Crude confidence interval from weighted std
    var = sum(w * (d - weighted_mean) ** 2 for d, w in zip(deltas, weights)) / total_w
    sd = math.sqrt(max(0.0, var))

    return {
        "elo_est": round(weighted_mean, 1),
        "elo_low": round(weighted_mean - 1.96 * sd, 1),
        "elo_high": round(weighted_mean + 1.96 * sd, 1),
        "per_match_elo_est": [round(d, 1) for d in deltas],
        "method": "inverse-variance weighted (across skill levels)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to Petra model .pt")
    ap.add_argument("--stockfish", required=True, help="Path to stockfish binary")
    ap.add_argument("--skill-levels", type=int, nargs="+", default=[0, 5, 10],
                    help="Stockfish skill levels to match against")
    ap.add_argument("--games", type=int, default=100,
                    help="Games per skill level (must be even — half W/half B)")
    ap.add_argument("--n-sim", type=int, default=100,
                    help="MCTS simulations per move for Petra")
    ap.add_argument("--think-time-ms", type=int, default=100,
                    help="Stockfish think time per move (ms)")
    ap.add_argument("--out", default=None, help="JSON output path (optional)")
    args = ap.parse_args()

    if args.games % 2 != 0:
        print(f"Warning: --games {args.games} is odd; rounding up to "
              f"{args.games + 1} to balance W/B.")
        args.games += 1

    print("=" * 64)
    print("PETRA ELO BENCHMARK vs Stockfish-Skill")
    print("=" * 64)
    print(f"  Model        : {args.model}")
    print(f"  Stockfish    : {args.stockfish}")
    print(f"  Skill levels : {args.skill_levels}")
    print(f"  Games each   : {args.games} (half W / half B)")
    print(f"  Petra n_sim  : {args.n_sim}")
    print(f"  SF think (ms): {args.think_time_ms}")
    print(f"  Date         : {datetime.date.today()}")
    print("=" * 64)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device} ...")
    model = PetraNet().to(device)
    sd = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"  Loaded {args.model}")
    print()

    petra = MCTSAgent(model, n_simulations=args.n_sim, value="learned",
                      temperature_moves=10)

    results = []
    for skill in args.skill_levels:
        elo = SKILL_TO_ELO[skill]
        print(f"--- skill={skill}  (Stockfish ≈ {elo} ELO) ---")
        r = play_match(petra, args.stockfish, skill, args.games,
                       args.think_time_ms)
        results.append(r)
        print(f"  Result: W={r['wins']} D={r['draws']} L={r['losses']}  "
              f"wr={r['wr']:.3f}  (vs SF~{elo})")
        print()

    print("=" * 64)
    print("ELO ESTIMATE")
    print("=" * 64)
    elo = estimate_elo(results)
    print(f"  Estimated Petra ELO : {elo['elo_est']}")
    if elo.get("elo_low") is not None:
        print(f"  ~95% CI             : [{elo['elo_low']}, {elo['elo_high']}]")
    print(f"  Per-match ELOs      : {elo.get('per_match_elo_est')}")
    print(f"  Method              : {elo['method']}")
    print("=" * 64)

    if args.out:
        out = {
            "model":       args.model,
            "stockfish":   args.stockfish,
            "n_sim":       args.n_sim,
            "think_ms":    args.think_time_ms,
            "date":        str(datetime.date.today()),
            "matches":     results,
            "elo_est":     elo,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
