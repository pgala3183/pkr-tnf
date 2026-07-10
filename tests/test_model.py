"""Tests for the poker action transformer."""

import pytest
import torch

from poker_transformer.model.transformer import ModelConfig, PokerTransformer, load_model_config


@pytest.fixture
def config() -> ModelConfig:
    return load_model_config()


@pytest.fixture
def model(config: ModelConfig) -> PokerTransformer:
    return PokerTransformer(config)


def test_config_matches_yaml(config: ModelConfig) -> None:
    assert config.n_layer == 6
    assert config.n_head == 8
    assert config.n_embd == 256
    assert config.block_size == 256
    assert config.dropout == 0.1
    assert config.vocab_size == 235


def test_forward_pass_output_shapes(model: PokerTransformer, config: ModelConfig) -> None:
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    action_logits, win_prob = model(input_ids)

    assert action_logits.shape == (batch_size, seq_len, config.vocab_size)
    assert win_prob.shape == (batch_size, 1)
    assert torch.isfinite(action_logits).all()
    assert torch.isfinite(win_prob).all()
    assert ((win_prob >= 0) & (win_prob <= 1)).all()


def test_forward_pass_max_block_size(model: PokerTransformer, config: ModelConfig) -> None:
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size))
    action_logits, win_prob = model(input_ids)

    assert action_logits.shape == (2, config.block_size, config.vocab_size)
    assert win_prob.shape == (2, 1)


def test_sequence_longer_than_block_size_raises(model: PokerTransformer, config: ModelConfig) -> None:
    input_ids = torch.randint(0, config.vocab_size, (1, config.block_size + 1))
    with pytest.raises(ValueError, match="block_size"):
        model(input_ids)


def test_causal_attention_is_decoder_only(model: PokerTransformer, config: ModelConfig) -> None:
    """Changing a future token must not change logits at earlier positions."""
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        logits_before, _ = model(input_ids)

        modified = input_ids.clone()
        modified[0, -1] = (modified[0, -1] + 1) % config.vocab_size
        logits_after, _ = model(modified)

    assert torch.allclose(logits_before[0, :-1, :], logits_after[0, :-1, :])
