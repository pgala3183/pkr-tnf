"""End-to-end test: demo transformer bot plays a hand against the ONNX API."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

DEMO_ROOT = Path(__file__).resolve().parents[1] / "demo"
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

# Stub flask_socketio before demo app imports (not required for generator test).
if "flask_socketio" not in sys.modules:
    flask_socketio_stub = types.ModuleType("flask_socketio")

    class _SocketIO:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            pass

    flask_socketio_stub.SocketIO = _SocketIO
    flask_socketio_stub.emit = lambda *args, **kwargs: None
    sys.modules["flask_socketio"] = flask_socketio_stub

from poker_transformer.model.transformer import PokerTransformer, load_model_config
from poker_transformer.serving.export_onnx import export_and_quantize
from poker_transformer.serving.inference import OnnxPredictor


@pytest.fixture(scope="module")
def onnx_predictor(tmp_path_factory: pytest.TempPathFactory) -> OnnxPredictor:
    output_dir = tmp_path_factory.mktemp("demo_onnx")
    config = load_model_config()
    model = PokerTransformer(config)
    checkpoint = tmp_path_factory.mktemp("demo_ckpt") / "best.pt"
    torch.save(
        {"step": 0, "model_state_dict": model.state_dict(), "model_config": config.__dict__},
        checkpoint,
    )
    export_and_quantize(checkpoint, output_dir)
    return OnnxPredictor(output_dir / "model.int8.onnx")


def test_transformer_demo_hand_completes(monkeypatch: pytest.MonkeyPatch, onnx_predictor: OnnxPredictor) -> None:
    """Simulate a full heads-up hand with transformer bot via mocked HTTP + socket emits."""

    import httpx

    def fake_post(url: str, json: dict, timeout: float = 10.0):
        del url, timeout
        result = onnx_predictor.predict(json)
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = MagicMock(
            return_value={
                "action": result.action,
                "amount": result.amount,
                "action_probabilities": result.action_probabilities,
                "win_probability": result.win_probability,
            }
        )
        return response

    monkeypatch.setattr(httpx, "post", fake_post)

    from app.poker_game import game

    gen = game("transformer", "test_player_sid")
    human_actions = [["call"], ["check"], ["check"], ["check"], ["check"]]
    action_index = 0
    max_steps = 200
    steps = 0
    finished = False

    state = next(gen)
    while steps < max_steps and not finished:
        steps += 1
        if state == "wait_for_player_decision":
            if action_index >= len(human_actions):
                decision = ["call"]
            else:
                decision = human_actions[action_index]
                action_index += 1
            state = gen.send(decision)
        elif state == "end":
            finished = True
        else:
            state = next(gen)

    assert finished, "Game generator should reach 'end' state"
    assert steps < max_steps, "Hand should finish within step budget"
