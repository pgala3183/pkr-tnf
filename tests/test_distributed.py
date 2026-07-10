"""Smoke tests for distributed training helpers (CPU/gloo, no CUDA required)."""

from __future__ import annotations

import os
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from poker_transformer.model.transformer import load_model_config
from poker_transformer.model.transformer import PokerTransformer
from poker_transformer.training.distributed import unwrap_model


def _ddp_worker(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    cfg = load_model_config()
    model = PokerTransformer(cfg)
    model = DDP(model)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, win_prob = model(x)
    loss = logits.mean() + win_prob.mean()
    loss.backward()

    # All ranks should have identical grads after DDP all-reduce averaging.
    grad_norm = next(unwrap_model(model).parameters()).grad.norm().item()
    tensor = torch.tensor([grad_norm])
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    assert gathered[0].item() == gathered[-1].item()

    dist.destroy_process_group()


def test_ddp_gradient_sync_cpu() -> None:
    world_size = 2
    mp.spawn(_ddp_worker, args=(world_size,), nprocs=world_size, join=True)


def test_unwrap_model() -> None:
    cfg = load_model_config()
    base = PokerTransformer(cfg)
    wrapped = mock.Mock()
    wrapped.module = base
    assert unwrap_model(wrapped) is base
    assert unwrap_model(base) is base
