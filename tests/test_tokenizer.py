"""Tests for the heads-up NLHE action tokenizer."""

from pathlib import Path

import pytest

from poker_transformer.tokenizer.encode import (
    decode_token,
    encode_action,
    encode_hand_start,
    encode_special,
    encode_token_for_roundtrip,
    size_bucket_label,
)
from poker_transformer.tokenizer.vocab import Vocabulary


@pytest.fixture
def vocab() -> Vocabulary:
    return Vocabulary.from_config()


def test_vocabulary_size_in_target_range(vocab: Vocabulary) -> None:
    assert 150 <= vocab.size <= 250


def test_round_trip_every_vocab_token(vocab: Vocabulary) -> None:
    for token_id, token in enumerate(vocab.tokens):
        encoded_id = encode_token_for_roundtrip(token, vocab)
        assert encoded_id == token_id
        assert decode_token(encoded_id, vocab) == token


def test_vocab_json_round_trip(vocab: Vocabulary, tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    vocab.save_json(path)
    loaded = Vocabulary.load_json(path)
    assert loaded.tokens == vocab.tokens
    assert loaded.size == vocab.size


def test_preflop_sb_open_raise(vocab: Vocabulary) -> None:
    game_state = {
        "street": "PREFLOP",
        "position": "SB",
        "pot": 30,
        "big_blind": 20,
        "hero_stack": 1000,
        "villain_stack": 1000,
    }
    action_dict = {"action_type": "RAISE", "amount": 60}

    token_id = encode_action(action_dict, game_state, vocab)
    assert decode_token(token_id, vocab) == "PREFLOP|SB|RAISE|200-300%"


def test_flop_bb_check(vocab: Vocabulary) -> None:
    game_state = {
        "street": "FLOP",
        "position": "BB",
        "pot": 120,
        "big_blind": 20,
        "hero_stack": 940,
        "villain_stack": 940,
    }
    action_dict = {"action_type": "CHECK"}

    token_id = encode_action(action_dict, game_state, vocab)
    assert decode_token(token_id, vocab) == "FLOP|BB|CHECK"


def test_river_sb_fold(vocab: Vocabulary) -> None:
    game_state = {
        "street": "RIVER",
        "position": "SB",
        "pot": 400,
        "big_blind": 20,
        "hero_stack": 300,
        "villain_stack": 500,
    }
    action_dict = {"action_type": "FOLD"}

    token_id = encode_action(action_dict, game_state, vocab)
    assert decode_token(token_id, vocab) == "RIVER|SB|FOLD"


def test_hand_start_short_stack(vocab: Vocabulary) -> None:
    game_state = {"hero_stack": 150, "villain_stack": 180, "big_blind": 20}
    token_id = encode_hand_start(game_state, vocab)
    assert decode_token(token_id, vocab) == "HAND_START|0-10bb"


def test_special_tokens(vocab: Vocabulary) -> None:
    assert decode_token(encode_special("HAND_END", vocab), vocab) == "HAND_END"
    assert decode_token(encode_special("SHOWDOWN", vocab), vocab) == "SHOWDOWN"
    assert decode_token(encode_special("PAD", vocab), vocab) == "<PAD>"


def test_example_hand_sequence(vocab: Vocabulary) -> None:
    """Short heads-up hand: open, call, check-check, bet-call."""
    sequence = []

    sequence.append(
        encode_hand_start(
            {"hero_stack": 1000, "villain_stack": 1000, "big_blind": 20},
            vocab,
        )
    )
    sequence.append(
        encode_action(
            {"action_type": "RAISE", "amount": 50},
            {
                "street": "PREFLOP",
                "position": "SB",
                "pot": 30,
                "big_blind": 20,
                "hero_stack": 1000,
                "villain_stack": 1000,
            },
            vocab,
        )
    )
    sequence.append(
        encode_action(
            {"action_type": "CALL", "amount": 40},
            {
                "street": "PREFLOP",
                "position": "BB",
                "pot": 80,
                "big_blind": 20,
                "hero_stack": 950,
                "villain_stack": 960,
            },
            vocab,
        )
    )
    sequence.append(
        encode_action(
            {"action_type": "CHECK"},
            {
                "street": "FLOP",
                "position": "SB",
                "pot": 120,
                "big_blind": 20,
                "hero_stack": 950,
                "villain_stack": 960,
            },
            vocab,
        )
    )
    sequence.append(
        encode_action(
            {"action_type": "CHECK"},
            {
                "street": "FLOP",
                "position": "BB",
                "pot": 120,
                "big_blind": 20,
                "hero_stack": 950,
                "villain_stack": 960,
            },
            vocab,
        )
    )
    sequence.append(
        encode_action(
            {"action_type": "BET", "amount": 60},
            {
                "street": "TURN",
                "position": "SB",
                "pot": 120,
                "big_blind": 20,
                "hero_stack": 950,
                "villain_stack": 960,
            },
            vocab,
        )
    )
    sequence.append(
        encode_action(
            {"action_type": "CALL", "amount": 60},
            {
                "street": "TURN",
                "position": "BB",
                "pot": 180,
                "big_blind": 20,
                "hero_stack": 890,
                "villain_stack": 960,
            },
            vocab,
        )
    )
    sequence.append(encode_special("SHOWDOWN", vocab))
    sequence.append(encode_special("HAND_END", vocab))

    decoded = [decode_token(token_id, vocab) for token_id in sequence]
    assert decoded == [
        "HAND_START|40-60bb",
        "PREFLOP|SB|RAISE|150-200%",
        "PREFLOP|BB|CALL",
        "FLOP|SB|CHECK",
        "FLOP|BB|CHECK",
        "TURN|SB|BET|40-60%",
        "TURN|BB|CALL",
        "SHOWDOWN",
        "HAND_END",
    ]


def test_size_bucket_label_midpoints(vocab: Vocabulary) -> None:
    game_state = {"pot": 100, "big_blind": 20}
    assert size_bucket_label({"action_type": "BET", "amount": 5}, game_state, vocab) == "0-10%"
    assert size_bucket_label({"action_type": "BET", "amount": 17}, game_state, vocab) == "10-25%"
    assert size_bucket_label({"action_type": "ALL_IN", "amount": 999}, game_state, vocab) == "ALL_IN"


if __name__ == "__main__":
    vocabulary = Vocabulary.from_config()
    print(f"Vocabulary size: {vocabulary.size}")
