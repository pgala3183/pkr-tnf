"""Smoke test: vendored PyPokerEngine runs a full heads-up game."""

import pypokerengine
from pypokerengine.api.game import setup_config, start_poker

from poker_transformer.engine_integration.fish_player import FishPlayer


def test_engine_smoke():
    # 1. Import pypokerengine (module-level imports above)
    assert pypokerengine is not None

    # 2. Register two FishPlayer instances (always calls)
    config = setup_config(max_round=10, initial_stack=1000, small_blind_amount=20)
    config.register_player(name="fish_1", algorithm=FishPlayer())
    config.register_player(name="fish_2", algorithm=FishPlayer())

    # 3. Run a 10-round game
    game_result = start_poker(config, verbose=0)

    # 4. Assert valid game_result with stack info for both players
    assert isinstance(game_result, dict)
    assert "players" in game_result
    assert len(game_result["players"]) == 2

    for player in game_result["players"]:
        assert isinstance(player, dict)
        assert "name" in player
        assert "stack" in player
        assert isinstance(player["stack"], int)

    names = {player["name"] for player in game_result["players"]}
    assert names == {"fish_1", "fish_2"}
