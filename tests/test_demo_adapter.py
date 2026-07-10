"""Tests for demo <-> /predict adapter translation."""

from __future__ import annotations

from poker_transformer.serving.demo_adapter import (
    DemoActionRecord,
    DemoHandState,
    build_predict_payload,
    demo_options_to_valid_actions,
    predict_response_to_demo_decision,
    street_from_common_cards,
)


def test_street_from_board_length() -> None:
    assert street_from_common_cards(None) == "preflop"
    assert street_from_common_cards(["AS", "KD", "2C"]) == "flop"
    assert street_from_common_cards(["AS", "KD", "2C", "5H"]) == "turn"


def test_demo_options_to_valid_actions_check_raise() -> None:
    valid = demo_options_to_valid_actions(
        {"fold": True, "check": True, "call": False, "raise": True},
        call_value=0,
        min_raise=50,
        max_raise=1000,
    )
    assert {"action": "fold", "amount": 0} in valid
    assert {"action": "call", "amount": 0} in valid
    assert any(item["action"] == "raise" for item in valid)


def test_build_predict_payload_roundtrip_fields() -> None:
    state = DemoHandState(
        street="preflop",
        hero_position="BB",
        action_history=[
            DemoActionRecord(
                street="PREFLOP",
                position="SB",
                action_type="CALL",
                amount=25,
                pot_before=1,
                hero_stack=1000,
                villain_stack=1000,
            )
        ],
        hole_cards=["AS", "KD"],
        valid_actions=[{"action": "call", "amount": 0}],
        pot_size=75,
        hero_stack=975,
        villain_stack=975,
        big_blind=50,
        initial_hero_stack=1000,
        initial_villain_stack=1000,
    )
    payload = build_predict_payload(state)
    assert payload["street"] == "preflop"
    assert payload["position"] == "BB"
    assert payload["action_history"][0]["action_type"] == "CALL"
    assert payload["hole_cards"] == ["AS", "KD"]


def test_predict_response_to_demo_decision() -> None:
    assert predict_response_to_demo_decision({"action": "fold", "amount": 0}) == ["fold"]
    assert predict_response_to_demo_decision({"action": "call", "amount": 0}) == ["check"]
    assert predict_response_to_demo_decision({"action": "call", "amount": 50}) == ["call"]
    assert predict_response_to_demo_decision({"action": "raise", "amount": 100}) == ["raise", 100]
