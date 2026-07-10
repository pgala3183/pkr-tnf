"""Integration test: TransformerPlayer completes a full game vs FishPlayer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pypokerengine.api.game import setup_config, start_poker

from poker_transformer.engine_integration.fish_player import FishPlayer
from poker_transformer.engine_integration.transformer_player import TransformerPlayer
from poker_transformer.model.transformer import PokerTransformer, load_model_config


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    """Untrained checkpoint with valid structure for smoke testing."""
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


def test_transformer_player_vs_fish_completes_game(checkpoint_path: Path) -> None:
    config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=20)
    config.register_player(
        name="transformer",
        algorithm=TransformerPlayer(checkpoint_path, policy="greedy", device="cpu"),
    )
    config.register_player(name="fish", algorithm=FishPlayer())

    game_result = start_poker(config, verbose=0)

    assert isinstance(game_result, dict)
    assert "players" in game_result
    assert len(game_result["players"]) == 2

    for player in game_result["players"]:
        assert isinstance(player, dict)
        assert "name" in player
        assert "stack" in player
        assert isinstance(player["stack"], int)

    names = {player["name"] for player in game_result["players"]}
    assert names == {"transformer", "fish"}


def test_transformer_player_sampled_policy(checkpoint_path: Path) -> None:
    config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=20)
    config.register_player(
        name="transformer",
        algorithm=TransformerPlayer(
            checkpoint_path,
            policy="sampled",
            temperature=1.0,
            device="cpu",
            seed=42,
        ),
    )
    config.register_player(name="fish", algorithm=FishPlayer())

    game_result = start_poker(config, verbose=0)
    assert len(game_result["players"]) == 2
