"""PyTorch Dataset and DataLoader utilities for self-play shards."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from poker_transformer.tokenizer.vocab import Vocabulary


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    win_labels: torch.Tensor


def load_hands(data_dir: str | Path) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    hands: list[dict[str, Any]] = []
    for shard_path in sorted(data_dir.glob("self_play_*.pt")):
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        hands.extend(payload["hands"])
    hands.sort(key=lambda hand: int(hand["hand_id"]))
    return hands


def split_hands(
    hands: list[dict[str, Any]],
    *,
    val_ratio: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not hands:
        raise ValueError("No hands found for train/val split")

    val_size = max(1, int(len(hands) * val_ratio))
    train_hands = hands[:-val_size]
    val_hands = hands[-val_size:]
    return train_hands, val_hands


class SelfPlayDataset(Dataset):
    """One training example per hand (shared token sequence + win label for bot_a)."""

    def __init__(
        self,
        hands: list[dict[str, Any]],
        *,
        perspective: str = "bot_a",
    ) -> None:
        self.hands = hands
        self.perspective = perspective

    def __len__(self) -> int:
        return len(self.hands)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        hand = self.hands[index]
        perspectives = hand.get("player_perspectives") or {}
        perspective = perspectives.get(self.perspective) or {}
        token_ids = perspective.get("token_ids", hand["token_ids"])
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        else:
            token_ids = token_ids.to(dtype=torch.long)

        winners = hand.get("result", {}).get("winners", [])
        win_label = 1.0 if self.perspective in winners else 0.0

        return {
            "token_ids": token_ids,
            "win_label": torch.tensor(win_label, dtype=torch.float32),
        }


def collate_hands(
    batch: list[dict[str, torch.Tensor]],
    *,
    pad_id: int,
    block_size: int,
) -> Batch:
    batch_size = len(batch)
    max_len = block_size

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
    win_labels = torch.stack([item["win_label"] for item in batch])

    for index, item in enumerate(batch):
        sequence = item["token_ids"]
        if sequence.size(0) > max_len:
            sequence = sequence[-max_len:]
        length = sequence.size(0)
        input_ids[index, -length:] = sequence
        attention_mask[index, -length:] = 1.0

    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        win_labels=win_labels,
    )


def make_dataloader(
    hands: list[dict[str, Any]],
    *,
    vocab: Vocabulary,
    batch_size: int,
    block_size: int,
    shuffle: bool,
    num_workers: int = 0,
    sampler: DistributedSampler | None = None,
) -> DataLoader:
    dataset = SelfPlayDataset(hands)

    def _collate(batch: list[dict[str, torch.Tensor]]) -> Batch:
        return collate_hands(batch, pad_id=vocab.pad_id, block_size=block_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
