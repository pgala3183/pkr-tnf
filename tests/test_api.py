"""Tests for the FastAPI prediction service."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from poker_transformer.model.transformer import PokerTransformer, load_model_config
from poker_transformer.serving.api import create_app
from poker_transformer.serving.export_onnx import export_and_quantize


@pytest.fixture(scope="module")
def onnx_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export a smoke fp32+int8 ONNX model once for API tests."""
    output_dir = tmp_path_factory.mktemp("onnx")
    config = load_model_config()
    model = PokerTransformer(config)
    checkpoint = tmp_path_factory.mktemp("ckpt") / "best.pt"
    torch.save(
        {
            "step": 0,
            "model_state_dict": model.state_dict(),
            "model_config": config.__dict__,
        },
        checkpoint,
    )
    export_and_quantize(checkpoint, output_dir)
    return output_dir


@pytest.fixture
def client(onnx_model_dir: Path) -> TestClient:
    int8_path = onnx_model_dir / "model.int8.onnx"
    app = create_app(int8_path)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_path"].endswith("model.int8.onnx")


def test_predict_sample_hand(client: TestClient) -> None:
    payload = {
        "street": "preflop",
        "position": "BB",
        "hole_cards": ["Ah", "Kd"],
        "action_history": [
            {
                "street": "PREFLOP",
                "position": "SB",
                "action_type": "CALL",
                "amount": 20,
                "pot_before": 30,
                "hero_stack": 980,
                "villain_stack": 990,
            },
            {
                "street": "PREFLOP",
                "position": "BB",
                "action_type": "CHECK",
                "amount": 0,
                "pot_before": 50,
                "hero_stack": 980,
                "villain_stack": 980,
            },
        ],
        "valid_actions": [
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 0},
            {"action": "raise", "amount": {"min": 40, "max": 980}},
        ],
        "pot_size": 40,
        "hero_stack": 980,
        "villain_stack": 980,
        "big_blind": 20,
        "initial_hero_stack": 1000,
        "initial_villain_stack": 1000,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()

    assert body["action"] in {"fold", "call", "raise"}
    assert isinstance(body["amount"], int)
    assert body["amount"] >= 0
    assert 0.0 <= body["win_probability"] <= 1.0

    assert isinstance(body["action_probabilities"], list)
    assert len(body["action_probabilities"]) >= 1
    for entry in body["action_probabilities"]:
        assert "token" in entry
        assert "action_type" in entry
        assert 0.0 <= entry["probability"] <= 1.0

    prob_sum = sum(entry["probability"] for entry in body["action_probabilities"])
    assert abs(prob_sum - 1.0) < 1e-4


def test_predict_rejects_invalid_raise_amount(client: TestClient) -> None:
    payload = {
        "street": "preflop",
        "position": "SB",
        "valid_actions": [
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 20},
            {"action": "raise", "amount": 40},
        ],
        "pot_size": 30,
        "hero_stack": 980,
        "villain_stack": 960,
        "big_blind": 20,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
