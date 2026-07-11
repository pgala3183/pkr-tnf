"""Heads-up bb/100 roll-outs of TransformerPlayer vs baseline bots.

Each "hand" is an independent 1-round PyPokerEngine game with fresh stacks so
bust-outs do not truncate the sample. Hero seat (SB/BB) is alternated so
position bias cancels out.

bb/100 = mean(chip_delta / big_blind) * 100
95% CI uses the normal approximation: mean ± z * stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDORED_ENGINE_ROOT = PROJECT_ROOT / "src" / "engine_integration" / "pypokerengine"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "results" / "baseline_eval.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "postln" / "best.pt"

OpponentName = Literal["FishPlayer", "HonestPlayer", "RandomPlayer"]


def _ensure_engine_path() -> None:
    vendored_root = str(VENDORED_ENGINE_ROOT)
    if vendored_root not in sys.path:
        sys.path.insert(0, vendored_root)


_ensure_engine_path()

from examples.players.honest_player import HonestPlayer  # noqa: E402
from examples.players.random_player import RandomPlayer  # noqa: E402
from pypokerengine.api.game import setup_config, start_poker  # noqa: E402
from pypokerengine.players import BasePokerPlayer  # noqa: E402

from poker_transformer.engine_integration.fish_player import FishPlayer  # noqa: E402
from poker_transformer.engine_integration.transformer_player import TransformerPlayer  # noqa: E402


class FastHonestPlayer(HonestPlayer):
    """HonestPlayer with fewer Monte Carlo sims for faster eval."""

    def declare_action(self, valid_actions, hole_card, round_state):
        from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards

        community_card = round_state["community_card"]
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=50,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card),
        )
        if win_rate >= 1.0 / self.nb_player:
            action = valid_actions[1]
        else:
            action = valid_actions[0]
        return action["action"], action["amount"]


OPPONENT_DESCRIPTIONS: dict[OpponentName, str] = {
    "FishPlayer": "Always calls (PyPokerEngine tutorial bot)",
    "HonestPlayer": "Monte Carlo equity threshold bot",
    "RandomPlayer": "Uniform random legal action",
}


def make_opponent(name: OpponentName) -> BasePokerPlayer:
    if name == "FishPlayer":
        return FishPlayer()
    if name == "HonestPlayer":
        return FastHonestPlayer()
    if name == "RandomPlayer":
        return RandomPlayer()
    raise ValueError(f"Unknown opponent: {name}")


@dataclass
class MatchupResult:
    name: str
    description: str
    hands: int
    bb_per_100: float
    ci_low: float
    ci_high: float
    std_error: float
    mean_bb_per_hand: float
    std_bb_per_hand: float


def _player_stack(game_result: dict[str, Any], name: str) -> int:
    for player in game_result["players"]:
        if player["name"] == name:
            return int(player["stack"])
    raise KeyError(f"Player {name!r} missing from game result")


def play_one_hand(
    *,
    checkpoint: Path,
    opponent_name: OpponentName,
    initial_stack: int,
    small_blind: int,
    hero_is_first: bool,
    device: str,
    policy: str,
    seed: int | None,
) -> float:
    """Play one hand; return hero chip delta (positive = hero won chips)."""
    hero = TransformerPlayer(
        checkpoint,
        policy=policy,  # type: ignore[arg-type]
        device=device,
        seed=seed,
    )
    opponent = make_opponent(opponent_name)

    config = setup_config(
        max_round=1,
        initial_stack=initial_stack,
        small_blind_amount=small_blind,
    )
    if hero_is_first:
        config.register_player(name="transformer", algorithm=hero)
        config.register_player(name="opponent", algorithm=opponent)
    else:
        config.register_player(name="opponent", algorithm=opponent)
        config.register_player(name="transformer", algorithm=hero)

    result = start_poker(config, verbose=0)
    final_stack = _player_stack(result, "transformer")
    return float(final_stack - initial_stack)


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(var)


def summarize_hand_deltas(
    deltas: list[float],
    *,
    big_blind: int,
    confidence_level: float = 0.95,
    name: str,
    description: str,
) -> MatchupResult:
    bb_deltas = [d / big_blind for d in deltas]
    mean_bb, std_bb = _mean_std(bb_deltas)
    n = max(len(bb_deltas), 1)
    stderr = std_bb / math.sqrt(n)
    # Normal approx; z≈1.96 for 95%.
    z = 1.95996398454 if abs(confidence_level - 0.95) < 1e-9 else 1.95996398454
    bb_per_100 = mean_bb * 100.0
    half = z * stderr * 100.0
    return MatchupResult(
        name=name,
        description=description,
        hands=len(deltas),
        bb_per_100=bb_per_100,
        ci_low=bb_per_100 - half,
        ci_high=bb_per_100 + half,
        std_error=stderr * 100.0,
        mean_bb_per_hand=mean_bb,
        std_bb_per_hand=std_bb,
    )


def evaluate_matchup(
    checkpoint: Path,
    opponent: OpponentName,
    *,
    hands: int = 1000,
    initial_stack: int = 1000,
    small_blind: int = 10,
    big_blind: int | None = None,
    device: str = "cpu",
    policy: str = "greedy",
    seed: int = 0,
    confidence_level: float = 0.95,
    progress: bool = True,
) -> MatchupResult:
    """Run ``hands`` independent HU hands vs one opponent; return bb/100 + CI."""
    bb = big_blind if big_blind is not None else 2 * small_blind
    deltas: list[float] = []
    iterator: Any = range(hands)
    if progress:
        iterator = tqdm(iterator, desc=f"vs {opponent}", unit="hand")

    for hand_idx in iterator:
        delta = play_one_hand(
            checkpoint=checkpoint,
            opponent_name=opponent,
            initial_stack=initial_stack,
            small_blind=small_blind,
            hero_is_first=(hand_idx % 2 == 0),
            device=device,
            policy=policy,
            seed=None if seed < 0 else seed + hand_idx,
        )
        deltas.append(delta)

    return summarize_hand_deltas(
        deltas,
        big_blind=bb,
        confidence_level=confidence_level,
        name=opponent,
        description=OPPONENT_DESCRIPTIONS[opponent],
    )


def run_rollout_eval(
    checkpoint: Path,
    *,
    opponents: list[OpponentName] | None = None,
    hands_per_matchup: int = 1000,
    initial_stack: int = 1000,
    small_blind: int = 10,
    device: str = "cpu",
    policy: str = "greedy",
    seed: int = 0,
    confidence_level: float = 0.95,
    output_path: Path | None = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    opponents = opponents or ["FishPlayer", "HonestPlayer", "RandomPlayer"]
    big_blind = 2 * small_blind
    results = [
        evaluate_matchup(
            checkpoint,
            opponent,
            hands=hands_per_matchup,
            initial_stack=initial_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            device=device,
            policy=policy,
            seed=seed + 10_000 * i,
            confidence_level=confidence_level,
        )
        for i, opponent in enumerate(opponents)
    ]

    report: dict[str, Any] = {
        "eval_date": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint).replace("\\", "/"),
        "hands_per_matchup": hands_per_matchup,
        "big_blind": big_blind,
        "small_blind": small_blind,
        "starting_stack_bb": initial_stack / big_blind,
        "confidence_level": confidence_level,
        "policy": policy,
        "device": device,
        "seed": seed,
        "opponents": [
            {
                "name": r.name,
                "description": r.description,
                "bb_per_100": round(r.bb_per_100, 3),
                "ci_low": round(r.ci_low, 3),
                "ci_high": round(r.ci_high, 3),
                "std_error": round(r.std_error, 3),
                "hands": r.hands,
            }
            for r in results
        ],
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="bb/100 roll-out eval vs baseline bots.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Transformer checkpoint (.pt)",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=1000,
        help="Hands per opponent (use 10000 for README-quality CIs)",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["FishPlayer", "HonestPlayer", "RandomPlayer"],
        choices=["FishPlayer", "HonestPlayer", "RandomPlayer"],
    )
    parser.add_argument("--initial-stack", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=10, help="SB chips; BB = 2×SB")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--policy", choices=["greedy", "sampled"], default="greedy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path (default: eval/results/baseline_eval.json)",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    report = run_rollout_eval(
        args.checkpoint,
        opponents=list(args.opponents),  # type: ignore[arg-type]
        hands_per_matchup=args.hands,
        initial_stack=args.initial_stack,
        small_blind=args.small_blind,
        device=args.device,
        policy=args.policy,
        seed=args.seed,
        confidence_level=args.confidence_level,
        output_path=args.output,
    )

    print("\n=== bb/100 roll-out ===")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
