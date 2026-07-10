"""Benchmark PyTorch vs Triton fused residual+LayerNorm and plot speedup."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from poker_transformer.model.kernels.fused_residual_layernorm import (
    fused_residual_layernorm,
    fused_residual_layernorm_ref,
    triton_is_available,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "results" / "kernel_benchmark.png"

# Shapes representative of poker-transformer training batches, plus larger
# batches to show where kernel fusion amortizes launch overhead.
BENCHMARK_SHAPES: list[tuple[int, int, int]] = [
    (4, 32, 256),
    (8, 64, 256),
    (16, 128, 256),
    (32, 256, 256),
    (64, 256, 256),
    (128, 256, 256),
]

WARMUP_ITERS = 20
TIMED_ITERS = 100


def _time_fn(
    fn,
    *args,
    warmup: int = WARMUP_ITERS,
    iters: int = TIMED_ITERS,
    **kwargs,
) -> float:
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3  # ms per iter


def run_benchmark(
    shapes: list[tuple[int, int, int]] | None = None,
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    if not triton_is_available():
        raise RuntimeError("CUDA and Triton are required to run the kernel benchmark")

    import matplotlib.pyplot as plt

    shapes = shapes or BENCHMARK_SHAPES
    labels: list[str] = []
    pytorch_ms: list[float] = []
    triton_ms: list[float] = []
    speedups: list[float] = []

    device = torch.device("cuda")
    eps = 1e-5

    for batch_size, seq_len, n_embd in shapes:
        x = torch.randn(batch_size, seq_len, n_embd, device=device)
        residual = torch.randn_like(x)
        weight = torch.randn(n_embd, device=device)
        bias = torch.randn(n_embd, device=device)

        pt_ms = _time_fn(
            fused_residual_layernorm_ref, x, residual, weight, bias, eps
        )
        tr_ms = _time_fn(
            fused_residual_layernorm, x, residual, weight, bias, eps, use_triton=True
        )

        label = f"B={batch_size}\nT={seq_len}"
        labels.append(label)
        pytorch_ms.append(pt_ms)
        triton_ms.append(tr_ms)
        speedups.append(pt_ms / tr_ms)

        print(f"{label.replace(chr(10), ' ')}: PyTorch {pt_ms:.3f} ms, Triton {tr_ms:.3f} ms, "
              f"speedup {speedups[-1]:.2f}x")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_pos = range(len(labels))
    width = 0.35

    fig, (ax_time, ax_speed) = plt.subplots(1, 2, figsize=(12, 5))

    ax_time.bar([i - width / 2 for i in x_pos], pytorch_ms, width, label="PyTorch")
    ax_time.bar([i + width / 2 for i in x_pos], triton_ms, width, label="Triton")
    ax_time.set_xticks(list(x_pos))
    ax_time.set_xticklabels(labels)
    ax_time.set_ylabel("Time (ms / forward)")
    ax_time.set_title("Fused residual + LayerNorm")
    ax_time.legend()

    ax_speed.bar(list(x_pos), speedups, color="seagreen")
    ax_speed.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax_speed.set_xticks(list(x_pos))
    ax_speed.set_xticklabels(labels)
    ax_speed.set_ylabel("Speedup (PyTorch / Triton)")
    ax_speed.set_title("Triton speedup")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved benchmark plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the speedup bar chart PNG",
    )
    args = parser.parse_args()
    run_benchmark(output_path=args.output)


if __name__ == "__main__":
    main()
