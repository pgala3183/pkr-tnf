"""Encode poker actions to token ids and decode back to readable strings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from poker_transformer.tokenizer.vocab import (
    DEFAULT_CONFIG_PATH,
    Vocabulary,
    load_config,
)

DEFAULT_VOCAB_JSON = Path(__file__).resolve().parents[2] / "data" / "processed" / "vocab.json"

_SIZEless = frozenset({"FOLD", "CHECK", "CALL", "ALL_IN"})


def _effective_stack_bb(game_state: dict[str, Any]) -> float:
    hero_stack = float(game_state["hero_stack"])
    villain_stack = float(game_state["villain_stack"])
    big_blind = float(game_state["big_blind"])
    if big_blind <= 0:
        raise ValueError("big_blind must be positive")
    return min(hero_stack, villain_stack) / big_blind


def bucket_value(
    value: float,
    buckets: tuple[dict[str, Any], ...],
) -> str:
    """Return the label for the bucket containing ``value``."""
    if value == float("inf"):
        for bucket in buckets:
            if bucket["low"] is None and bucket["high"] is None:
                return bucket["label"]
        return buckets[-1]["label"]

    for index, bucket in enumerate(buckets):
        low = bucket["low"]
        high = bucket["high"]
        is_last = index == len(buckets) - 1

        if low is None and high is None:
            continue

        lower_ok = True if low is None else value >= float(low)
        upper_ok = True
        if high is not None:
            upper_ok = value < float(high) if not is_last else value <= float(high)

        if lower_ok and upper_ok:
            return bucket["label"]

    return buckets[-1]["label"]


def stack_bucket_label(game_state: dict[str, Any], vocab: Vocabulary) -> str:
    return bucket_value(_effective_stack_bb(game_state), vocab.config.stack_buckets)


def pot_relative_size_ratio(action_dict: dict[str, Any], game_state: dict[str, Any]) -> float:
    pot = float(game_state.get("pot", 0))
    amount = float(action_dict.get("amount", 0))
    if pot <= 0:
        return float("inf")
    return amount / pot


def size_bucket_label(action_dict: dict[str, Any], game_state: dict[str, Any], vocab: Vocabulary) -> str:
    if action_dict["action_type"] == "ALL_IN":
        return "ALL_IN"
    ratio = pot_relative_size_ratio(action_dict, game_state)
    return bucket_value(ratio, vocab.config.size_buckets)


def _require_game_state_fields(game_state: dict[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in game_state]
    if missing:
        raise KeyError(f"game_state missing required fields: {missing}")


def encode_hand_start(game_state: dict[str, Any], vocab: Vocabulary | None = None) -> int:
    vocab = vocab or Vocabulary.from_config()
    _require_game_state_fields(game_state, ("hero_stack", "villain_stack", "big_blind"))
    label = stack_bucket_label(game_state, vocab)
    token = f"HAND_START|{label}"
    return vocab.id_for(token)


def encode_special(token_name: str, vocab: Vocabulary | None = None) -> int:
    vocab = vocab or Vocabulary.from_config()
    if token_name == "PAD":
        return vocab.id_for(vocab.config.pad_token)
    if token_name not in vocab.config.special_tokens:
        raise ValueError(f"Unknown special token: {token_name}")
    return vocab.id_for(token_name)


def encode_action(
    action_dict: dict[str, Any],
    game_state: dict[str, Any],
    vocab: Vocabulary | None = None,
) -> int:
    """Encode one betting event to a vocabulary token id."""
    vocab = vocab or Vocabulary.from_config()
    _require_game_state_fields(game_state, ("street", "position", "pot", "big_blind"))

    action_type = action_dict["action_type"].upper()
    if action_type not in vocab.config.action_types:
        raise ValueError(f"Unsupported action_type: {action_type}")

    street = game_state["street"].upper()
    position = game_state["position"].upper()
    if street not in vocab.config.streets:
        raise ValueError(f"Unsupported street: {street}")
    if position not in vocab.config.positions:
        raise ValueError(f"Unsupported position: {position}")

    if action_type in _SIZEless:
        token = f"{street}|{position}|{action_type}"
    elif action_type in {"BET", "RAISE"}:
        size_label = size_bucket_label(action_dict, game_state, vocab)
        token = f"{street}|{position}|{action_type}|{size_label}"
    else:
        raise ValueError(f"Unhandled action_type: {action_type}")

    return vocab.id_for(token)


def decode_token(token_id: int, vocab: Vocabulary | None = None) -> str:
    """Decode a token id to its canonical vocabulary string."""
    vocab = vocab or Vocabulary.from_config()
    if token_id < 0 or token_id >= vocab.size:
        raise IndexError(f"token_id out of range: {token_id}")
    return vocab.token_for(token_id)


def encode_token_for_roundtrip(token: str, vocab: Vocabulary | None = None) -> int:
    """Build a valid encoding for any vocabulary token (used in tests)."""
    vocab = vocab or Vocabulary.from_config()
    parsed = vocab.parse_token(token)

    if parsed["kind"] == "pad":
        return encode_special("PAD", vocab)

    if parsed["kind"] == "special":
        return encode_special(parsed["token"], vocab)

    if parsed["kind"] == "hand_start":
        label = parsed["stack_bucket"]
        stack_bb = _stack_bb_for_label(label, vocab)
        game_state = _base_game_state(stack_bb)
        return encode_hand_start(game_state, vocab)

    game_state = {
        "street": parsed["street"],
        "position": parsed["position"],
        "pot": 100.0,
        "big_blind": 20.0,
        "hero_stack": 1000.0,
        "villain_stack": 1000.0,
    }
    action_dict: dict[str, Any] = {"action_type": parsed["action_type"]}

    if "size_bucket" in parsed:
        if parsed["size_bucket"] == "ALL_IN":
            action_dict["amount"] = float("inf")
        else:
            action_dict["amount"] = _amount_for_size_label(
                parsed["size_bucket"],
                game_state["pot"],
            )

    return encode_action(action_dict, game_state, vocab)


def _stack_bb_for_label(label: str, vocab: Vocabulary) -> float:
    for bucket in vocab.config.stack_buckets:
        if bucket["label"] == label:
            low = bucket["low"]
            high = bucket["high"]
            if low is None:
                return 5.0
            if high is None:
                return float(low) + 10.0
            return (float(low) + float(high)) / 2.0
    raise ValueError(f"Unknown stack bucket label: {label}")


def _amount_for_size_label(label: str, pot: float) -> float:
    if label == "500%+":
        return pot * 6.0
    if label.endswith("%"):
        text = label.replace("%", "")
        if "+" in text:
            return pot * 5.5
        if "-" in text:
            low_text, high_text = text.split("-", 1)
            midpoint = (float(low_text) + float(high_text)) / 200.0
            return pot * midpoint
    raise ValueError(f"Unknown size bucket label: {label}")


def _base_game_state(stack_bb: float) -> dict[str, float]:
    big_blind = 20.0
    stack = stack_bb * big_blind
    return {
        "hero_stack": stack,
        "villain_stack": stack,
        "big_blind": big_blind,
    }
