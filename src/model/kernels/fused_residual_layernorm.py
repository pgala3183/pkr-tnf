"""Triton fused residual-add + LayerNorm kernel and PyTorch references."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def fused_residual_layernorm_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """PyTorch reference: LayerNorm(x + residual)."""
    y = x + residual
    return F.layer_norm(y, (y.shape[-1],), weight, bias, eps)


def layernorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """PyTorch reference: LayerNorm(x)."""
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_residual_layernorm_kernel(
        x_ptr,
        residual_ptr,
        y_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        rstd_ptr,
        stride_row,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # program_id(axis): which program instance runs along this axis.
        # axis=0 → one Triton program per matrix row (token position).
        row = tl.program_id(0)

        # arange(0, BLOCK_SIZE): column indices handled by this program.
        col_offsets = tl.arange(0, BLOCK_SIZE)
        # Masking: when n_cols is not divisible by BLOCK_SIZE, ignore extra lanes.
        mask = col_offsets < n_cols

        row_x = x_ptr + row * stride_row
        row_r = residual_ptr + row * stride_row
        row_y = y_ptr + row * stride_row

        x = tl.load(row_x + col_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(row_r + col_offsets, mask=mask, other=0.0).to(tl.float32)
        x = x + residual

        mean = tl.sum(x, axis=0) / n_cols
        x_centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(x_centered * x_centered, axis=0) / n_cols
        rstd = tl.rsqrt(var + eps)

        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        y = x_centered * rstd * weight + bias

        tl.store(row_y + col_offsets, y, mask=mask)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, rstd)

    @triton.jit
    def _layernorm_kernel(
        x_ptr,
        y_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        rstd_ptr,
        stride_row,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        row_x = x_ptr + row * stride_row
        row_y = y_ptr + row * stride_row

        x = tl.load(row_x + col_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / n_cols
        x_centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(x_centered * x_centered, axis=0) / n_cols
        rstd = tl.rsqrt(var + eps)

        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        y = x_centered * rstd * weight + bias

        tl.store(row_y + col_offsets, y, mask=mask)
        tl.store(mean_ptr + row, mean)
        tl.store(rstd_ptr + row, rstd)


def _block_size_for(n_cols: int) -> int:
    if TRITON_AVAILABLE:
        return max(triton.next_power_of_2(n_cols), 16)
    return max(n_cols, 16)


def _launch_fused_kernel(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert TRITON_AVAILABLE and x.is_cuda

    x_2d = x.reshape(-1, x.shape[-1])
    residual_2d = residual.reshape(-1, residual.shape[-1])
    n_rows, n_cols = x_2d.shape

    y = torch.empty_like(x_2d)
    mean = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)

    block_size = _block_size_for(n_cols)
    # grid: how many programs to launch per axis. (n_rows,) → one program per row.
    grid = (n_rows,)

    _fused_residual_layernorm_kernel[grid](
        x_2d,
        residual_2d,
        y,
        weight,
        bias,
        mean,
        rstd,
        x_2d.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
    )
    return y.view_as(x)


def _launch_layernorm_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    assert TRITON_AVAILABLE and x.is_cuda

    x_2d = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape
    y = torch.empty_like(x_2d)
    mean = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)

    block_size = _block_size_for(n_cols)
    grid = (n_rows,)

    _layernorm_kernel[grid](
        x_2d,
        y,
        weight,
        bias,
        mean,
        rstd,
        x_2d.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
    )
    return y.view_as(x)


class FusedResidualLayerNormFn(torch.autograd.Function):
    """Triton forward, PyTorch reference backward for training correctness."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        use_triton: bool,
    ) -> torch.Tensor:
        ctx.eps = eps
        ctx.use_triton = use_triton and TRITON_AVAILABLE and x.is_cuda

        if ctx.use_triton:
            y = _launch_fused_kernel(x, residual, weight, bias, eps)
        else:
            y = fused_residual_layernorm_ref(x, residual, weight, bias, eps)

        ctx.save_for_backward(x, residual, weight, bias)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, residual, weight, bias = ctx.saved_tensors
        x_req = x.detach().requires_grad_(True)
        residual_req = residual.detach().requires_grad_(True)
        weight_req = weight.detach().requires_grad_(True)
        bias_req = bias.detach().requires_grad_(True)

        with torch.enable_grad():
            y = fused_residual_layernorm_ref(
                x_req,
                residual_req,
                weight_req,
                bias_req,
                ctx.eps,
            )
            grad_x, grad_residual, grad_weight, grad_bias = torch.autograd.grad(
                y,
                (x_req, residual_req, weight_req, bias_req),
                grad_output,
                retain_graph=False,
            )

        return grad_x, grad_residual, grad_weight, grad_bias, None, None


class LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        use_triton: bool,
    ) -> torch.Tensor:
        ctx.eps = eps
        ctx.use_triton = use_triton and TRITON_AVAILABLE and x.is_cuda

        if ctx.use_triton:
            y = _launch_layernorm_kernel(x, weight, bias, eps)
        else:
            y = layernorm_ref(x, weight, bias, eps)

        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, bias = ctx.saved_tensors
        x_req = x.detach().requires_grad_(True)
        weight_req = weight.detach().requires_grad_(True)
        bias_req = bias.detach().requires_grad_(True)

        with torch.enable_grad():
            y = layernorm_ref(x_req, weight_req, bias_req, ctx.eps)
            grad_x, grad_weight, grad_bias = torch.autograd.grad(
                y,
                (x_req, weight_req, bias_req),
                grad_output,
                retain_graph=False,
            )

        return grad_x, grad_weight, grad_bias, None, None


def fused_residual_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    return FusedResidualLayerNormFn.apply(x, residual, weight, bias, eps, use_triton)


def triton_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    return LayerNormFn.apply(x, weight, bias, eps, use_triton)


def triton_is_available() -> bool:
    return TRITON_AVAILABLE and torch.cuda.is_available()
