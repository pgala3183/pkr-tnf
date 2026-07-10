"""Decoder-only transformer for next-action prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from poker_transformer.model.kernels.fused_residual_layernorm import (
    fused_residual_layernorm,
    triton_layernorm,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    dropout: float
    vocab_size: int
    mlp_ratio: int = 4
    use_triton_kernels: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        return cls(
            n_layer=int(raw["n_layer"]),
            n_head=int(raw["n_head"]),
            n_embd=int(raw["n_embd"]),
            block_size=int(raw["block_size"]),
            dropout=float(raw["dropout"]),
            vocab_size=int(raw["vocab_size"]),
            mlp_ratio=int(raw.get("mlp_ratio", 4)),
            use_triton_kernels=bool(raw.get("use_triton_kernels", False)),
        )

    @property
    def head_size(self) -> int:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        return self.n_embd // self.n_head


def load_model_config(config_path: str | Path | None = None) -> ModelConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ModelConfig.from_dict(raw)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (decoder-only)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_size = config.head_size
        self.n_embd = config.n_embd

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        shape = (batch_size, seq_len, self.n_head, self.head_size)
        q = q.view(*shape).transpose(1, 2)
        k = k.view(*shape).transpose(1, 2)
        v = v.view(*shape).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn = attn.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        out = self.proj(out)
        return self.resid_dropout(out)


class MLP(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.fc = nn.Linear(config.n_embd, hidden)
        self.proj = nn.Linear(hidden, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Decoder block with optional Triton fused residual+LayerNorm (Post-LN path)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.use_triton_kernels = config.use_triton_kernels
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton_kernels:
            # Post-LN fused path: LayerNorm(x + sublayer(x)) in one Triton kernel.
            x = fused_residual_layernorm(
                x,
                self.attn(x),
                self.ln1.weight,
                self.ln1.bias,
                self.ln1.eps,
                use_triton=True,
            )
            x = fused_residual_layernorm(
                x,
                self.mlp(x),
                self.ln2.weight,
                self.ln2.bias,
                self.ln2.eps,
                use_triton=True,
            )
            return x

        # Default Pre-LN PyTorch path (original implementation).
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PokerTransformer(nn.Module):
    """
    Decoder-only transformer for poker action sequences.

    - Action head: next-token logits at every position.
    - Value head: win-probability scalar from the final hidden state.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.action_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, 1),
        )

        self.apply(self._init_weights)

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "PokerTransformer":
        return cls(load_model_config(config_path))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: Long tensor of shape (batch, seq_len) with token ids.

        Returns:
            action_logits: (batch, seq_len, vocab_size)
            win_prob: (batch, 1) sigmoid win-probability estimate
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, seq_len)")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        if self.config.use_triton_kernels:
            hidden = triton_layernorm(
                x,
                self.ln_f.weight,
                self.ln_f.bias,
                self.ln_f.eps,
                use_triton=True,
            )
        else:
            hidden = self.ln_f(x)
        action_logits = self.action_head(hidden)
        win_prob = torch.sigmoid(self.value_head(hidden[:, -1, :]))

        if return_hidden_states:
            return action_logits, win_prob, hidden
        return action_logits, win_prob
