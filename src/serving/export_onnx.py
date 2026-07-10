"""Export a trained poker-transformer checkpoint to ONNX (fp32 + dynamic int8)."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

from poker_transformer.model.transformer import ModelConfig, PokerTransformer
from poker_transformer.training.evaluate_loss import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "onnx"

FP32_NAME = "model.fp32.onnx"
INT8_NAME = "model.int8.onnx"

WARMUP_PASSES = 20
TIMED_PASSES = 200
ATOL = 1e-4
RTOL = 1e-3
ONNX_OPSET = 17


class OnnxExportModel(nn.Module):
    """Thin wrapper so torch.onnx.export sees stable input/output names."""

    def __init__(self, model: PokerTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_logits, win_prob = self.model(input_ids)
        return action_logits, win_prob


def _onnx_safe_config(model_cfg: ModelConfig) -> ModelConfig:
    """Triton custom ops are not exportable; force the PyTorch Pre-LN path."""
    if model_cfg.use_triton_kernels:
        return replace(model_cfg, use_triton_kernels=False)
    return model_cfg


def load_model_for_export(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[PokerTransformer, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = _onnx_safe_config(ModelConfig(**payload["model_config"]))
    model = PokerTransformer(model_cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, payload


def make_test_inputs(
    *,
    vocab_size: int,
    block_size: int,
    batch_size: int = 2,
    num_cases: int = 5,
    seed: int = 0,
) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    seq_lengths = np.linspace(4, min(block_size, 64), num=num_cases, dtype=int)
    inputs: list[torch.Tensor] = []
    for seq_len in seq_lengths:
        ids = rng.integers(0, vocab_size, size=(batch_size, int(seq_len)), dtype=np.int64)
        inputs.append(torch.from_numpy(ids))
    return inputs


def export_fp32_onnx(
    model: PokerTransformer,
    output_path: Path,
    *,
    opset: int = ONNX_OPSET,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = OnnxExportModel(model).cpu()
    wrapper.eval()

    config = model.config
    dummy = torch.zeros(1, min(16, config.block_size), dtype=torch.long)

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "action_logits": {0: "batch", 1: "seq_len"},
        "win_prob": {0: "batch"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            input_names=["input_ids"],
            output_names=["action_logits", "win_prob"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)


def validate_onnx_matches_pytorch(
    model: PokerTransformer,
    onnx_path: Path,
    test_inputs: list[torch.Tensor],
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    model = model.cpu()
    model.eval()

    for index, input_ids in enumerate(test_inputs):
        with torch.no_grad():
            pt_logits, pt_win_prob = model(input_ids)
        ort_outputs = session.run(
            None,
            {input_name: input_ids.numpy()},
        )
        ort_logits, ort_win_prob = ort_outputs

        assert np.allclose(pt_logits.numpy(), ort_logits, atol=atol, rtol=rtol), (
            f"action_logits mismatch on test input {index}"
        )
        assert np.allclose(pt_win_prob.numpy(), ort_win_prob, atol=atol, rtol=rtol), (
            f"win_prob mismatch on test input {index}"
        )


def quantize_onnx_model(fp32_path: Path, int8_path: Path) -> None:
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QUInt8,
    )
    onnx.checker.check_model(onnx.load(str(int8_path)))


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def benchmark_onnx_session(
    session: ort.InferenceSession,
    test_inputs: list[np.ndarray],
    *,
    warmup: int = WARMUP_PASSES,
    iters: int = TIMED_PASSES,
) -> tuple[float, float]:
    input_name = session.get_inputs()[0].name

    for _ in range(warmup):
        for sample in test_inputs:
            session.run(None, {input_name: sample})

    latencies_ms: list[float] = []
    cycle = 0
    while len(latencies_ms) < iters:
        sample = test_inputs[cycle % len(test_inputs)]
        cycle += 1
        start = time.perf_counter()
        session.run(None, {input_name: sample})
        latencies_ms.append((time.perf_counter() - start) * 1_000.0)

    latencies_ms.sort()
    mean_ms = statistics.mean(latencies_ms)
    p95_index = min(len(latencies_ms) - 1, int(0.95 * len(latencies_ms)))
    p95_ms = latencies_ms[p95_index]
    return mean_ms, p95_ms


def export_and_quantize(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    opset: int = ONNX_OPSET,
    seed: int = 0,
) -> dict[str, float | str]:
    device = torch.device("cpu")
    model, payload = load_model_for_export(checkpoint_path, device)
    test_inputs = make_test_inputs(
        vocab_size=model.config.vocab_size,
        block_size=model.config.block_size,
        seed=seed,
    )
    numpy_inputs = [sample.numpy() for sample in test_inputs]

    fp32_path = output_dir / FP32_NAME
    int8_path = output_dir / INT8_NAME

    export_fp32_onnx(model, fp32_path, opset=opset)
    validate_onnx_matches_pytorch(model, fp32_path, test_inputs)

    quantize_onnx_model(fp32_path, int8_path)

    fp32_session = ort.InferenceSession(
        str(fp32_path),
        providers=["CPUExecutionProvider"],
    )
    int8_session = ort.InferenceSession(
        str(int8_path),
        providers=["CPUExecutionProvider"],
    )

    fp32_mean_ms, fp32_p95_ms = benchmark_onnx_session(fp32_session, numpy_inputs)
    int8_mean_ms, int8_p95_ms = benchmark_onnx_session(int8_session, numpy_inputs)

    report = {
        "checkpoint": str(checkpoint_path),
        "fp32_path": str(fp32_path),
        "int8_path": str(int8_path),
        "step": payload.get("step"),
        "fp32_size_mb": _file_size_mb(fp32_path),
        "int8_size_mb": _file_size_mb(int8_path),
        "size_reduction_pct": 100.0 * (1.0 - _file_size_mb(int8_path) / _file_size_mb(fp32_path)),
        "fp32_mean_latency_ms": fp32_mean_ms,
        "fp32_p95_latency_ms": fp32_p95_ms,
        "int8_mean_latency_ms": int8_mean_ms,
        "int8_p95_latency_ms": int8_p95_ms,
        "latency_speedup_mean": fp32_mean_ms / int8_mean_ms if int8_mean_ms > 0 else float("inf"),
    }
    return report


def print_report(report: dict[str, float | str]) -> None:
    print("\n=== ONNX export summary ===")
    print(f"Checkpoint: {report['checkpoint']} (step {report.get('step')})")
    print(f"FP32 model: {report['fp32_path']}")
    print(f"INT8 model: {report['int8_path']}")
    print(
        f"Model size: {report['fp32_size_mb']:.2f} MB -> {report['int8_size_mb']:.2f} MB "
        f"({report['size_reduction_pct']:.1f}% smaller)"
    )
    print(
        f"FP32 latency: mean {report['fp32_mean_latency_ms']:.3f} ms, "
        f"p95 {report['fp32_p95_latency_ms']:.3f} ms"
    )
    print(
        f"INT8 latency: mean {report['int8_mean_latency_ms']:.3f} ms, "
        f"p95 {report['int8_p95_latency_ms']:.3f} ms "
        f"({report['latency_speedup_mean']:.2f}x mean speedup)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exported ONNX models",
    )
    parser.add_argument("--opset", type=int, default=ONNX_OPSET, help="ONNX opset version")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for validation inputs")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    report = export_and_quantize(
        args.checkpoint,
        args.output_dir,
        opset=args.opset,
        seed=args.seed,
    )
    print_report(report)


if __name__ == "__main__":
    main()
