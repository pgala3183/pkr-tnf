"""
Educational demo: hand-rolled int8 linear quantization vs ONNX Runtime.

We pick one layer (``blocks[0].mlp.fc``, the first MLP up-projection), quantize
its *weights* to int8 with explicit scale factors, and compare accuracy/speed
against fp32 PyTorch and ``onnxruntime.quantization.quantize_dynamic``.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic

from poker_transformer.model.transformer import PokerTransformer
from poker_transformer.tokenizer.vocab import Vocabulary
from poker_transformer.training.data import load_hands, make_dataloader, split_hands
from poker_transformer.training.evaluate_loss import load_checkpoint
from poker_transformer.training.train import load_training_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best.pt"
INT8_MAX = 127.0  # symmetric int8 uses [-127, 127] to keep zero exactly representable

WARMUP = 30
TIMED = 300


@dataclass(frozen=True)
class QuantizedLinearWeights:
    """Container for a manually quantized nn.Linear weight matrix."""

    weight_int8: torch.Tensor  # int8, shape (out_features, in_features)
    scales: torch.Tensor  # float32, shape () for per-tensor or (out_features,) per-channel
    bias: torch.Tensor | None
    mode: str  # "per_tensor" or "per_channel"


@dataclass(frozen=True)
class ErrorStats:
    max_abs_diff: float
    mean_abs_diff: float
    num_samples: int


@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    p95_ms: float


def symmetric_scale_from_absmax(abs_max: torch.Tensor) -> torch.Tensor:
    """
    Compute the fp32->int8 scale from a (possibly per-channel) absolute maximum.

    scale = abs_max / 127 means dequant(weight_q) = weight_q * scale recovers
    the original magnitude. We divide by 127 (not 128) for symmetric quantization
    so that zero maps exactly and outliers do not clip as aggressively.
    """
    abs_max = abs_max.clamp(min=1e-8)
    return abs_max / INT8_MAX


def quantize_weight_per_tensor(weight: torch.Tensor) -> QuantizedLinearWeights:
    """
    Per-tensor quantization: ONE scale for the entire weight matrix.

    Simplest scheme — one float stored alongside the int8 blob — but every
    output channel must share the same scale, so small channels get rounded
    away when a large channel sets a wide dynamic range.
    """
    abs_max = weight.abs().max()
    scale = symmetric_scale_from_absmax(abs_max)
    weight_int8 = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
    return QuantizedLinearWeights(
        weight_int8=weight_int8,
        scales=scale,
        bias=None,
        mode="per_tensor",
    )


def quantize_weight_per_channel(weight: torch.Tensor) -> QuantizedLinearWeights:
    """
    Per-channel (per output row) quantization.

    nn.Linear stores weight with shape (out_features, in_features). Each output
    neuron gets its own scale computed from that row's min/max, which preserves
    small-magnitude filters that would be drowned out under a global scale.
    """
    # Reduce across input features -> one abs-max per output channel.
    abs_max = weight.abs().amax(dim=1)
    scales = symmetric_scale_from_absmax(abs_max)
    # Broadcast scales across columns so row i divides by scales[i].
    weight_int8 = torch.round(weight / scales.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
    return QuantizedLinearWeights(
        weight_int8=weight_int8,
        scales=scales,
        bias=None,
        mode="per_channel",
    )


def dequantize_weights(q: QuantizedLinearWeights) -> torch.Tensor:
    """Reconstruct fp32 weights from int8 codes and scales (dequantize-on-the-fly)."""
    if q.mode == "per_tensor":
        return q.weight_int8.float() * q.scales
    return q.weight_int8.float() * q.scales.unsqueeze(1)


def int8_linear_dequant_on_the_fly(
    x: torch.Tensor,
    q: QuantizedLinearWeights,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """
    Quantized matmul via dequantize-on-the-fly.

    Instead of a true int8 GEMM kernel, we:
      1. Promote int8 weights back to fp32 with their scales (cheap vs reading fp32 from DRAM)
      2. Call the regular fp32 linear / matmul

    Real runtimes fuse steps 1+2 inside specialized kernels; this version is
    easier to read while still showing where error enters.
    """
    weight_fp32 = dequantize_weights(q)
    return F.linear(x, weight_fp32, bias)


def fp32_linear(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    return F.linear(x, linear.weight, linear.bias)


def compute_error(reference: torch.Tensor, approximate: torch.Tensor) -> ErrorStats:
    diff = (reference - approximate).abs()
    return ErrorStats(
        max_abs_diff=float(diff.max().item()),
        mean_abs_diff=float(diff.mean().item()),
        num_samples=int(diff.numel()),
    )


@torch.no_grad()
def collect_fc_activations(
    model: PokerTransformer,
    dataloader,
    device: torch.device,
    *,
    max_tokens: int = 20_000,
) -> torch.Tensor:
    """
    Run validation batches and capture real inputs to ``blocks[0].mlp.fc``.

    We mask out padding tokens so statistics reflect real hand tokens only.
    """
    model.eval()
    captured: list[torch.Tensor] = []
    total = 0

    def pre_hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal total
        if total >= max_tokens:
            return
        x = inputs[0]
        # Hook fires during forward; we stash the full tensor and slice later.
        captured.append(x.detach().cpu())

    handle = model.blocks[0].mlp.fc.register_forward_pre_hook(pre_hook)

    for batch in dataloader:
        if total >= max_tokens:
            break
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        model(input_ids)

        # Keep only non-pad positions from the most recent capture.
        x = captured[-1]
        mask = attention_mask.cpu().bool()
        x_valid = x[mask]
        captured[-1] = x_valid
        total += x_valid.shape[0]

    handle.remove()

    activations = torch.cat(captured, dim=0)
    return activations[:max_tokens]


def benchmark_fn(fn, *args, warmup: int = WARMUP, iters: int = TIMED) -> LatencyStats:
    for _ in range(warmup):
        fn(*args)
    latencies: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn(*args)
        latencies.append((time.perf_counter() - start) * 1_000.0)
    latencies.sort()
    p95_idx = min(len(latencies) - 1, int(0.95 * len(latencies)))
    return LatencyStats(mean_ms=statistics.mean(latencies), p95_ms=latencies[p95_idx])


def export_single_linear_onnx(linear: nn.Linear, path: Path, sample: torch.Tensor) -> None:
    """Export an isolated nn.Linear for apples-to-apples ORT dynamic quantization."""

    class LinearWrapper(nn.Module):
        def __init__(self, layer: nn.Linear) -> None:
            super().__init__()
            self.layer = layer

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer(x)

    wrapper = LinearWrapper(linear.cpu()).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Export with 2D activations (num_tokens, in_features) — one token per row.
    sample_2d = sample.reshape(-1, sample.shape[-1])[: min(64, sample.shape[0])]
    torch.onnx.export(
        wrapper,
        sample_2d,
        str(path),
        input_names=["activations"],
        output_names=["output"],
        dynamic_axes={"activations": {0: "num_tokens"}, "output": {0: "num_tokens"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))


def ort_int8_linear(
    session: ort.InferenceSession,
    x: torch.Tensor,
) -> torch.Tensor:
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: x.numpy().astype(np.float32)})[0]
    return torch.from_numpy(output)


def compare_ort_dynamic_quantization(
    linear: nn.Linear,
    activations: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[ErrorStats, LatencyStats, Path, Path]:
    """
    Export the same layer, run ``quantize_dynamic`` (Step 8), and benchmark ORT.

    ORT's dynamic quantizer also keeps activations in fp32 and compresses
    MatMul/Gemm weights — the same high-level pattern as our hand-rolled demo.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fp32_path = tmp_path / "linear.fp32.onnx"
        int8_path = tmp_path / "linear.int8.onnx"

        export_single_linear_onnx(linear, fp32_path, activations[: min(64, len(activations))])
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QUInt8,
        )

        session = ort.InferenceSession(
            str(int8_path),
            providers=["CPUExecutionProvider"],
        )
        ort_out = ort_int8_linear(session, activations)
        error = compute_error(reference, ort_out)
        latency = benchmark_fn(ort_int8_linear, session, activations)
        return error, latency, fp32_path, int8_path


def run_demo(
    checkpoint_path: Path,
    *,
    max_tokens: int = 20_000,
    batch_size: int = 32,
) -> dict[str, object]:
    device = torch.device("cpu")
    model, _payload = load_checkpoint(checkpoint_path, device)
    linear: nn.Linear = model.blocks[0].mlp.fc

    train_cfg = load_training_config()
    vocab = Vocabulary.from_config()
    hands = load_hands(PROJECT_ROOT / train_cfg.data_dir)
    _train, val_hands = split_hands(hands, val_ratio=train_cfg.val_ratio)
    val_loader = make_dataloader(
        val_hands,
        vocab=vocab,
        batch_size=batch_size,
        block_size=train_cfg.block_size,
        shuffle=False,
    )

    activations = collect_fc_activations(
        model,
        val_loader,
        device,
        max_tokens=max_tokens,
    )

    # Reference fp32 outputs on real validation activations.
    with torch.no_grad():
        reference = fp32_linear(activations, linear.cpu())

    q_tensor = quantize_weight_per_tensor(linear.weight.data)
    q_channel = quantize_weight_per_channel(linear.weight.data)

    out_tensor = int8_linear_dequant_on_the_fly(activations, q_tensor, linear.bias)
    out_channel = int8_linear_dequant_on_the_fly(activations, q_channel, linear.bias)

    err_tensor = compute_error(reference, out_tensor)
    err_channel = compute_error(reference, out_channel)

    lat_fp32 = benchmark_fn(fp32_linear, activations, linear)
    lat_tensor = benchmark_fn(
        int8_linear_dequant_on_the_fly,
        activations,
        q_tensor,
        linear.bias,
    )
    lat_channel = benchmark_fn(
        int8_linear_dequant_on_the_fly,
        activations,
        q_channel,
        linear.bias,
    )

    err_ort, lat_ort, _fp32_onnx, _int8_onnx = compare_ort_dynamic_quantization(
        linear,
        activations,
        reference,
    )

    weight_bytes_fp32 = linear.weight.numel() * 4
    weight_bytes_int8_tensor = linear.weight.numel() * 1 + 4  # int8 blob + one fp32 scale
    weight_bytes_int8_channel = linear.weight.numel() * 1 + linear.weight.shape[0] * 4

    return {
        "layer": "blocks[0].mlp.fc",
        "weight_shape": tuple(linear.weight.shape),
        "num_activation_tokens": int(activations.shape[0]),
        "in_features": linear.in_features,
        "out_features": linear.out_features,
        "err_fp32_vs_per_tensor": err_tensor,
        "err_fp32_vs_per_channel": err_channel,
        "err_fp32_vs_ort_dynamic": err_ort,
        "latency_fp32": lat_fp32,
        "latency_per_tensor": lat_tensor,
        "latency_per_channel": lat_channel,
        "latency_ort_dynamic": lat_ort,
        "weight_bytes_fp32": weight_bytes_fp32,
        "weight_bytes_int8_per_tensor": weight_bytes_int8_tensor,
        "weight_bytes_int8_per_channel": weight_bytes_int8_channel,
        "per_tensor_scale": float(q_tensor.scales.item()),
        "per_channel_scale_min": float(q_channel.scales.min().item()),
        "per_channel_scale_max": float(q_channel.scales.max().item()),
    }


def print_report(report: dict[str, object]) -> None:
    err_t: ErrorStats = report["err_fp32_vs_per_tensor"]
    err_c: ErrorStats = report["err_fp32_vs_per_channel"]
    err_o: ErrorStats = report["err_fp32_vs_ort_dynamic"]
    lat_f: LatencyStats = report["latency_fp32"]
    lat_t: LatencyStats = report["latency_per_tensor"]
    lat_c: LatencyStats = report["latency_per_channel"]
    lat_o: LatencyStats = report["latency_ort_dynamic"]

    print("\n=== int8 quantization demo: blocks[0].mlp.fc ===")
    print(f"Weight shape: {report['weight_shape']}  tokens evaluated: {report['num_activation_tokens']}")
    print(
        f"Weight storage: fp32 {report['weight_bytes_fp32'] / 1024:.1f} KiB | "
        f"int8 per-tensor ~{report['weight_bytes_int8_per_tensor'] / 1024:.1f} KiB | "
        f"int8 per-channel ~{report['weight_bytes_int8_per_channel'] / 1024:.1f} KiB"
    )
    print(f"Per-tensor scale: {report['per_tensor_scale']:.6e}")
    print(
        f"Per-channel scales: min {report['per_channel_scale_min']:.6e}, "
        f"max {report['per_channel_scale_max']:.6e}"
    )

    print("\nOutput error vs fp32 (validation activations):")
    print(
        f"  Hand per-tensor : max {err_t.max_abs_diff:.6f}, mean {err_t.mean_abs_diff:.6f}"
    )
    print(
        f"  Hand per-channel: max {err_c.max_abs_diff:.6f}, mean {err_c.mean_abs_diff:.6f}"
    )
    print(
        f"  ORT quantize_dynamic: max {err_o.max_abs_diff:.6f}, mean {err_o.mean_abs_diff:.6f}"
    )

    print("\nLatency (single layer, CPU):")
    print(f"  fp32 PyTorch      : mean {lat_f.mean_ms:.3f} ms, p95 {lat_f.p95_ms:.3f} ms")
    print(f"  hand per-tensor   : mean {lat_t.mean_ms:.3f} ms, p95 {lat_t.p95_ms:.3f} ms")
    print(f"  hand per-channel  : mean {lat_c.mean_ms:.3f} ms, p95 {lat_c.p95_ms:.3f} ms")
    print(f"  ORT int8 dynamic  : mean {lat_o.mean_ms:.3f} ms, p95 {lat_o.p95_ms:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20_000,
        help="Validation tokens to collect for layer inputs",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    report = run_demo(
        args.checkpoint,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )
    print_report(report)


if __name__ == "__main__":
    main()
