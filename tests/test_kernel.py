"""Numerical correctness tests for Triton fused LayerNorm kernels."""

from __future__ import annotations

import pytest
import torch

from poker_transformer.model.kernels.fused_residual_layernorm import (
    fused_residual_layernorm,
    fused_residual_layernorm_ref,
    layernorm_ref,
    triton_is_available,
    triton_layernorm,
)

pytestmark = pytest.mark.skipif(
    not triton_is_available(),
    reason="CUDA and Triton are required for kernel tests",
)

N_EMBD = 256
EPS = 1e-5
ATOL = 1e-4
RTOL = 1e-4


def _random_ln_inputs(
    batch_size: int,
    seq_len: int,
    n_embd: int = N_EMBD,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, seq_len, n_embd, device=device, dtype=torch.float32)
    residual = torch.randn_like(x)
    weight = torch.randn(n_embd, device=device, dtype=torch.float32)
    bias = torch.randn(n_embd, device=device, dtype=torch.float32)
    return x, residual, weight, bias


@pytest.mark.parametrize("batch_size,seq_len", [(4, 32), (8, 64), (2, 256)])
def test_fused_residual_layernorm_forward(batch_size: int, seq_len: int) -> None:
    device = torch.device("cuda")
    x, residual, weight, bias = _random_ln_inputs(batch_size, seq_len, device=device)

    expected = fused_residual_layernorm_ref(x, residual, weight, bias, EPS)
    actual = fused_residual_layernorm(
        x, residual, weight, bias, EPS, use_triton=True
    )

    assert torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("batch_size,seq_len", [(4, 32), (8, 64), (2, 256)])
def test_layernorm_forward(batch_size: int, seq_len: int) -> None:
    device = torch.device("cuda")
    x, _, weight, bias = _random_ln_inputs(batch_size, seq_len, device=device)

    expected = layernorm_ref(x, weight, bias, EPS)
    actual = triton_layernorm(x, weight, bias, EPS, use_triton=True)

    assert torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)


def test_fused_residual_layernorm_backward() -> None:
    device = torch.device("cuda")
    x, residual, weight, bias = _random_ln_inputs(4, 32, device=device)

    for use_triton in (False, True):
        x_t = x.detach().clone().requires_grad_(True)
        residual_t = residual.detach().clone().requires_grad_(True)
        weight_t = weight.detach().clone().requires_grad_(True)
        bias_t = bias.detach().clone().requires_grad_(True)

        y = fused_residual_layernorm(
            x_t, residual_t, weight_t, bias_t, EPS, use_triton=use_triton
        )
        loss = y.square().mean()
        loss.backward()

        assert x_t.grad is not None
        assert residual_t.grad is not None
        assert weight_t.grad is not None
        assert bias_t.grad is not None
        assert torch.isfinite(x_t.grad).all()
        assert torch.isfinite(residual_t.grad).all()


def test_triton_and_pytorch_backward_match() -> None:
    device = torch.device("cuda")
    x, residual, weight, bias = _random_ln_inputs(4, 32, device=device)
    grad_output = torch.randn_like(x)

    def grads(use_triton: bool) -> tuple[torch.Tensor, ...]:
        x_t = x.detach().clone().requires_grad_(True)
        residual_t = residual.detach().clone().requires_grad_(True)
        weight_t = weight.detach().clone().requires_grad_(True)
        bias_t = bias.detach().clone().requires_grad_(True)
        y = fused_residual_layernorm(
            x_t, residual_t, weight_t, bias_t, EPS, use_triton=use_triton
        )
        torch.autograd.grad(y, (x_t, residual_t, weight_t, bias_t), grad_output)

    ref_grads = grads(use_triton=False)
    triton_grads = grads(use_triton=True)

    for ref, actual in zip(ref_grads, triton_grads, strict=True):
        assert torch.allclose(ref, actual, atol=ATOL, rtol=RTOL)
