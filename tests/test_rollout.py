"""Tests for bb/100 roll-out harness."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from poker_transformer.eval.rollout import (
    evaluate_matchup,
    play_one_hand,
    run_rollout_eval,
    summarize_hand_deltas,
)
from poker_transformer.model.transformer import PokerTransformer, load_model_config


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    config = load_model_config()
    model = PokerTransformer(config)
    path = tmp_path / "smoke.pt"
    torch.save(
        {
            "step": 0,
            "model_state_dict": model.state_dict(),
            "model_config": config.__dict__,
        },
        path,
    )
    return path


def test_summarize_hand_deltas_bb_per_100() -> None:
    # +20 chips with BB=20 → +1 bb/hand → +100 bb/100
    result = summarize_hand_deltas(
        [20.0, 20.0, 20.0, 20.0],
        big_blind=20,
        name="FishPlayer",
        description="test",
    )
    assert result.bb_per_100 == pytest.approx(100.0)
    assert result.hands == 4
    assert result.ci_low <= result.bb_per_100 <= result.ci_high


def test_play_one_hand_returns_delta(checkpoint_path: Path) -> None:
    delta = play_one_hand(
        checkpoint=checkpoint_path,
        opponent_name="FishPlayer",
        initial_stack=1000,
        small_blind=10,
        hero_is_first=True,
        device="cpu",
        policy="greedy",
        seed=0,
    )
    assert isinstance(delta, float)
    assert -1000 <= delta <= 1000


def test_evaluate_matchup_smoke(checkpoint_path: Path) -> None:
    result = evaluate_matchup(
        checkpoint_path,
        "FishPlayer",
        hands=4,
        device="cpu",
        seed=1,
        progress=False,
    )
    assert result.hands == 4
    assert result.name == "FishPlayer"
    assert result.ci_low <= result.bb_per_100 <= result.ci_high


def test_run_rollout_writes_json(checkpoint_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "baseline_eval.json"
    report = run_rollout_eval(
        checkpoint_path,
        opponents=["FishPlayer"],
        hands_per_matchup=2,
        device="cpu",
        seed=2,
        output_path=out,
    )
    assert out.exists()
    assert report["opponents"][0]["name"] == "FishPlayer"
    assert report["opponents"][0]["hands"] == 2
    assert "bb_per_100" in report["opponents"][0]
