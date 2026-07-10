"""Triton fused residual-add + LayerNorm kernel and PyTorch references.

Performance notes (why this version is fast at poker-transformer shapes)
------------------------------------------------------------------------
The first version of this kernel lost to cuDNN LayerNorm at every benchmark
shape (0.30-0.48x). Profiling showed it was *host-side launch-bound*, not
GPU-bound: at D=256 each row is only 1 KiB of work, so the fixed cost per
call dominated. Four fixes:

1. **Multi-row tiled programs.** Each Triton program now normalizes a
   (ROWS_PER_PROGRAM x BLOCK_SIZE) tile instead of a single row, so weight
   and bias are loaded once per tile and the grid shrinks by up to 16x.
2. **Autotuning.** ROWS_PER_PROGRAM / num_warps are picked per (n_rows,
   n_cols) via ``triton.autotune`` — the best config for B=4,T=32 is not the
   best for B=32,T=256.
3. **No dead outputs.** The old kernel wrote per-row mean/rstd buffers that
   the backward never read (backward recomputes through the PyTorch
   reference). Dropping them removes two ``torch.empty`` allocations and two
   global-memory stores per call.
4. **Inference fast path.** When no input requires grad, we skip the
   ``torch.autograd.Function`` machinery entirely and launch the kernel
   directly.

The backward pass still recomputes through the PyTorch reference: it keeps
training numerically identical to ``F.layer_norm`` autograd and keeps this
module simple. The Triton win is therefore a *forward/inference* win.
"""

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
    # Candidate tilings. Small grids amortize per-program setup; more warps
    # help once the tile is large enough to keep them busy.
    _LN_CONFIGS = [
        triton.Config({"ROWS_PER_PROGRAM": 1}, num_warps=2),
        triton.Config({"ROWS_PER_PROGRAM": 2}, num_warps=2),
        triton.Config({"ROWS_PER_PROGRAM": 4}, num_warps=4),
        triton.Config({"ROWS_PER_PROGRAM": 8}, num_warps=4),
        triton.Config({"ROWS_PER_PROGRAM": 8}, num_warps=8),
        triton.Config({"ROWS_PER_PROGRAM": 16}, num_warps=8),
    ]

    @triton.autotune(configs=_LN_CONFIGS, key=["n_rows", "n_cols"])
    @triton.jit
    def _fused_residual_layernorm_kernel(
        x_ptr,
        residual_ptr,
        y_ptr,
        weight_ptr,
        bias_ptr,
        stride_row,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        # Each program handles a (ROWS_PER_PROGRAM, BLOCK_SIZE) tile.
        pid = tl.program_id(0)
        rows = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
        cols = tl.arange(0, BLOCK_SIZE)
        row_mask = rows < n_rows
        col_mask = cols < n_cols
        mask = row_mask[:, None] & col_mask[None, :]

        offsets = rows[:, None] * stride_row + cols[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = x + residual

        # Row-wise LayerNorm over the tile.
        mean = tl.sum(x, axis=1) / n_cols
        centered = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(centered * centered, axis=1) / n_cols
        rstd = tl.rsqrt(var + eps)

        # weight/bias loaded once per tile (not once per row).
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * rstd[:, None] * weight[None, :] + bias[None, :]

        tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.autotune(configs=_LN_CONFIGS, key=["n_rows", "n_cols"])
    @triton.jit
    def _layernorm_kernel(
        x_ptr,
        y_ptr,
        weight_ptr,
        bias_ptr,
        stride_row,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
        cols = tl.arange(0, BLOCK_SIZE)
        row_mask = rows < n_rows
        col_mask = cols < n_cols
        mask = row_mask[:, None] & col_mask[None, :]

        offsets = rows[:, None] * stride_row + cols[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=1) / n_cols
        centered = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(centered * centered, axis=1) / n_cols
        rstd = tl.rsqrt(var + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * rstd[:, None] * weight[None, :] + bias[None, :]

        tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


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

    x_2d = x.contiguous().view(-1, x.shape[-1])
    residual_2d = residual.contiguous().view(-1, residual.shape[-1])
    n_rows, n_cols = x_2d.shape
    y = torch.empty_like(x_2d)

    block_size = _block_size_for(n_cols)

    def grid(meta):
        return (triton.cdiv(n_rows, meta["ROWS_PER_PROGRAM"]),)

    _fused_residual_layernorm_kernel[grid](
        x_2d,
        residual_2d,
        y,
        weight,
        bias,
        x_2d.stride(0),
        n_rows,
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

    x_2d = x.contiguous().view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape
    y = torch.empty_like(x_2d)

    block_size = _block_size_for(n_cols)

    def grid(meta):
        return (triton.cdiv(n_rows, meta["ROWS_PER_PROGRAM"]),)

    _layernorm_kernel[grid](
        x_2d,
        y,
        weight,
        bias,
        x_2d.stride(0),
        n_rows,
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


def _grad_needed(*tensors: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def fused_residual_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    # Inference fast path: skip autograd.Function overhead entirely.
    if not _grad_needed(x, residual, weight, bias):
        if use_triton and TRITON_AVAILABLE and x.is_cuda:
            return _launch_fused_kernel(x, residual, weight, bias, eps)
        return fused_residual_layernorm_ref(x, residual, weight, bias, eps)
    return FusedResidualLayerNormFn.apply(x, residual, weight, bias, eps, use_triton)


def triton_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    if not _grad_needed(x, weight, bias):
        if use_triton and TRITON_AVAILABLE and x.is_cuda:
            return _launch_layernorm_kernel(x, weight, bias, eps)
        return layernorm_ref(x, weight, bias, eps)
    return LayerNormFn.apply(x, weight, bias, eps, use_triton)


def triton_is_available() -> bool:
    return TRITON_AVAILABLE and torch.cuda.is_available()
