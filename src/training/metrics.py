"""Shared training metrics and loss computation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from poker_transformer.model.transformer import PokerTransformer
from poker_transformer.training.data import Batch


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    action_loss: torch.Tensor
    value_loss: torch.Tensor
    action_perplexity: float


def compute_losses(
    model: PokerTransformer,
    batch: Batch,
    *,
    pad_id: int,
    value_loss_weight: float,
) -> LossOutput:
    action_logits, win_prob = model(batch.input_ids)

    shift_logits = action_logits[:, :-1, :].contiguous()
    shift_labels = batch.input_ids[:, 1:].contiguous()
    shift_mask = batch.attention_mask[:, 1:].contiguous()

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=pad_id,
    ).view(shift_labels.size(0), shift_labels.size(1))

    masked = per_token_loss * shift_mask
    denom = shift_mask.sum().clamp_min(1.0)
    action_loss = masked.sum() / denom

    value_loss = F.binary_cross_entropy(
        win_prob.view(-1),
        batch.win_labels.view(-1),
    )

    total_loss = action_loss + value_loss_weight * value_loss
    perplexity = math.exp(min(action_loss.item(), 20.0))

    return LossOutput(
        total_loss=total_loss,
        action_loss=action_loss,
        value_loss=value_loss,
        action_perplexity=perplexity,
    )


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


@torch.no_grad()
def evaluate(
    model: PokerTransformer,
    dataloader,
    *,
    pad_id: int,
    value_loss_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total = AverageMeter()
    action = AverageMeter()
    value = AverageMeter()

    for batch in dataloader:
        batch = Batch(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            win_labels=batch.win_labels.to(device),
        )
        losses = compute_losses(
            model,
            batch,
            pad_id=pad_id,
            value_loss_weight=value_loss_weight,
        )
        batch_size = batch.input_ids.size(0)
        total.update(losses.total_loss.item(), batch_size)
        action.update(losses.action_loss.item(), batch_size)
        value.update(losses.value_loss.item(), batch_size)

    return {
        "total_loss": total.average,
        "action_loss": action.average,
        "value_loss": value.average,
        "action_perplexity": math.exp(min(action.average, 20.0)),
    }
