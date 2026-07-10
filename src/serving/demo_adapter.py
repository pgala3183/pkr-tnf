"""
Adapter between the jarczano demo web app and poker-transformer /predict API.

Keeps the demo Flask app decoupled from tokenizer/model internals: the demo
builds a ``DemoHandState`` from its own objects, this module maps it to the
FastAPI JSON schema, and maps the response back to the demo decision format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

StreetToken = Literal["PREFLOP", "FLOP", "TURN", "RIVER"]
PositionToken = Literal["SB", "BB"]
ActionToken = Literal["FOLD", "CHECK", "CALL", "BET", "RAISE", "ALL_IN"]
ApiStreet = Literal["preflop", "flop", "turn", "river"]


@dataclass
class DemoActionRecord:
    """One betting event from the demo engine, tokenizer-aligned."""

    street: StreetToken
    position: PositionToken
    action_type: ActionToken
    amount: float
    pot_before: float
    hero_stack: int | None = None
    villain_stack: int | None = None


@dataclass
class DemoHandState:
    """
    Hand snapshot from the demo app at the moment the transformer bot must act.

    Stacks and action_history always use the *bot's* perspective as hero
    (matching self-play training where bot_a is the modeled player).
    """

    street: ApiStreet
    hero_position: PositionToken
    action_history: list[DemoActionRecord] = field(default_factory=list)
    hole_cards: list[str] = field(default_factory=list)
    valid_actions: list[dict[str, Any]] = field(default_factory=list)
    pot_size: float = 0.0
    hero_stack: int = 0
    villain_stack: int = 0
    big_blind: int = 0
    initial_hero_stack: int = 0
    initial_villain_stack: int = 0


STREET_FROM_BOARD = {
    0: "preflop",
    3: "flop",
    4: "turn",
    5: "river",
}


def street_from_common_cards(common_cards: list[str] | None) -> ApiStreet:
    """Map demo board length to API street name."""
    length = 0 if common_cards is None else len(common_cards)
    return STREET_FROM_BOARD.get(length, "preflop")  # type: ignore[return-value]


def street_token_from_common_cards(common_cards: list[str] | None) -> StreetToken:
    api = street_from_common_cards(common_cards)
    return api.upper()  # type: ignore[return-value]


def demo_options_to_valid_actions(
    dict_options: dict[str, bool],
    call_value: int,
    min_raise: int,
    max_raise: int,
) -> list[dict[str, Any]]:
    """Convert demo auction options to /predict ``valid_actions`` list."""
    valid: list[dict[str, Any]] = []

    if dict_options.get("fold"):
        valid.append({"action": "fold", "amount": 0})

    if dict_options.get("check"):
        valid.append({"action": "call", "amount": 0})
    elif dict_options.get("call"):
        valid.append({"action": "call", "amount": int(call_value)})

    if dict_options.get("raise"):
        valid.append(
            {
                "action": "raise",
                "amount": {"min": int(min_raise), "max": int(max_raise)},
            }
        )

    return valid


def normalize_demo_decision(
    decision: str,
    chips: int,
    *,
    call_value: int,
    player_stack: int,
    aggression_on_street: set[str],
    street: StreetToken,
) -> ActionToken | None:
    """Map demo engine decision strings to tokenizer action types."""
    if decision == "fold":
        return "FOLD"
    if decision == "check":
        return "CHECK"
    if decision == "call":
        if chips >= player_stack > 0:
            return "ALL_IN"
        return "CALL"
    if decision == "all-in":
        return "ALL_IN"
    if decision == "raise":
        if chips >= player_stack > 0:
            return "ALL_IN"
        if street not in aggression_on_street:
            return "BET"
        return "RAISE"
    return None


def build_predict_payload(state: DemoHandState) -> dict[str, Any]:
    """Translate demo hand state to POST /predict JSON body."""
    return {
        "street": state.street,
        "position": state.hero_position,
        "action_history": [
            {
                "street": item.street,
                "position": item.position,
                "action_type": item.action_type,
                "amount": item.amount,
                "pot_before": item.pot_before,
                "hero_stack": item.hero_stack,
                "villain_stack": item.villain_stack,
            }
            for item in state.action_history
        ],
        "hole_cards": list(state.hole_cards),
        "valid_actions": state.valid_actions,
        "pot_size": state.pot_size,
        "hero_stack": state.hero_stack,
        "villain_stack": state.villain_stack,
        "big_blind": state.big_blind,
        "initial_hero_stack": state.initial_hero_stack,
        "initial_villain_stack": state.initial_villain_stack,
    }


def predict_response_to_demo_decision(response: dict[str, Any]) -> list[Any]:
    """
    Map /predict response to demo decision list: ``['check']``, ``['call']``,
    ``['fold']``, ``['raise', amount]``, or ``['all-in']``.
    """
    action = response["action"]
    amount = int(response["amount"])

    if action == "fold":
        return ["fold"]
    if action == "call":
        if amount == 0:
            return ["check"]
        return ["call"]
    if action == "raise":
        return ["raise", amount]
    raise ValueError(f"Unsupported predict action: {action}")
