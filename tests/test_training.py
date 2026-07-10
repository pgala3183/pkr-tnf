"""Tests for training data pipeline and loss computation."""

import torch

from poker_transformer.model.transformer import PokerTransformer, load_model_config
from poker_transformer.tokenizer.vocab import Vocabulary
from poker_transformer.training.data import SelfPlayDataset, collate_hands, split_hands
from poker_transformer.training.metrics import compute_losses


def _make_hand(hand_id: int, token_ids: list[int], winners: list[str]) -> dict:
    return {
        "hand_id": hand_id,
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "result": {"winners": winners, "showdown": bool(winners)},
    }


def test_split_hands_is_by_hand_id_order() -> None:
    hands = [_make_hand(i, [1, 2, 3], ["bot_a"]) for i in range(10)]
    train, val = split_hands(hands, val_ratio=0.1)
    assert len(train) == 9
    assert len(val) == 1
    assert val[0]["hand_id"] == 9


def test_collate_left_pads_to_block_size() -> None:
    vocab = Vocabulary.from_config()
    batch = [
        {"token_ids": torch.tensor([3, 4, 5], dtype=torch.long), "win_label": torch.tensor(1.0)},
        {"token_ids": torch.tensor([7, 8], dtype=torch.long), "win_label": torch.tensor(0.0)},
    ]
    collated = collate_hands(batch, pad_id=vocab.pad_id, block_size=8)

    assert collated.input_ids.shape == (2, 8)
    assert collated.input_ids[0, -3:].tolist() == [3, 4, 5]
    assert collated.input_ids[1, -2:].tolist() == [7, 8]
    assert collated.input_ids[0, 0].item() == vocab.pad_id
    assert collated.attention_mask[0, -3:].tolist() == [1.0, 1.0, 1.0]
    assert collated.attention_mask[0, 0].item() == 0.0


def test_compute_losses_shapes() -> None:
    vocab = Vocabulary.from_config()
    model_cfg = load_model_config()
    model = PokerTransformer(model_cfg)

    hands = [_make_hand(0, [3, 11, 12, 9], ["bot_a"])]
    dataset = SelfPlayDataset(hands)
    sample = dataset[0]
    batch = collate_hands([sample], pad_id=vocab.pad_id, block_size=model_cfg.block_size)

    losses = compute_losses(model, batch, pad_id=vocab.pad_id, value_loss_weight=0.5)
    assert losses.total_loss.ndim == 0
    assert losses.action_loss.ndim == 0
    assert losses.value_loss.ndim == 0
    assert torch.isfinite(losses.total_loss)
