"""Eval package: engine roll-outs and bb/100 benchmarking."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poker_transformer.eval.rollout import evaluate_matchup, run_rollout_eval

__all__ = ["evaluate_matchup", "run_rollout_eval"]


def __getattr__(name: str):
    if name in ("evaluate_matchup", "run_rollout_eval"):
        from poker_transformer.eval import rollout

        return getattr(rollout, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
