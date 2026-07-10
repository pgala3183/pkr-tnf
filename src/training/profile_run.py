"""
Profile a short poker-transformer training run with torch.profiler.

Compute-bound vs memory-bound (how to read the output)
-------------------------------------------------------
* Compute-bound: the GPU spends most of its time in math kernels (matmul,
  attention, layer norm). The profiler's top CUDA ops will be things like
  `mm`, `bmm`, `softmax`, `layer_norm`, and CUDA time will dominate wall
  clock. Achieved FLOPs/sec will be a meaningful fraction of the GPU's
  advertised peak. Fix: bigger batch, larger model, mixed precision, kernel
  fusion — anything that keeps SMs busy.

* Memory-bound: the GPU waits on HBM bandwidth (loads/stores, copies,
  elementwise ops over large tensors). Top CUDA ops are often `memcpy`,
  `index`, `contiguous`, `fill_`, or many small kernels with low arithmetic
  intensity. Achieved FLOPs/sec will be far below peak even though the GPU
  is "busy". Fix: reduce padding waste, pin memory / faster DataLoader,
  gradient checkpointing tradeoffs, avoid unnecessary `.contiguous()` /
  host transfers.

Quick heuristic from this script's output:
1. Sort by `cuda_time_total` — if matmul/attention tops the list → lean
   compute-bound; if memcpy/index/elementwise dominate → lean memory-bound.
2. Compare achieved FLOPs/sec to peak — single-digit % with matmul on top
   often means short sequences / small batch (under-utilization), not pure
   memory bandwidth limits.
3. Check `profile_memory` in the Chrome trace — large allocation churn or
   idle gaps between kernels → memory / launch overhead.

Open the Chrome trace: chrome://tracing → Load → `logs/profiler/trace.json`
The same JSON file can also be opened in Perfetto (https://ui.perfetto.dev/).
For TensorBoard, symlink or copy `trace.json` into a logdir and use the
PyTorch profiler plugin, or re-run with `--export-tensorboard` (future).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

from poker_transformer.model.transformer import PokerTransformer, load_model_config
from poker_transformer.tokenizer.vocab import Vocabulary
from poker_transformer.training.data import Batch, load_hands, make_dataloader, set_seed, split_hands
from poker_transformer.training.metrics import compute_losses
from poker_transformer.training.train import TrainingConfig, learning_rate_at_step, load_training_config, set_optimizer_lr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"

# Advertised dense FP16/BF16 tensor-core peaks (FLOPs/s), approximate.
GPU_PEAK_FP16_FLOPS: dict[str, float] = {
    "A100": 312e12,
    "V100": 125e12,
    "T4": 65e12,
    "L4": 121e12,
    "RTX 4090": 82.6e12,
    "RTX 4080": 48.7e12,
    "RTX 3090": 35.6e12,
}


def detect_gpu_peak_flops(device_name: str) -> tuple[str, float | None]:
    for key, peak in GPU_PEAK_FP16_FLOPS.items():
        if key.lower() in device_name.lower():
            return key, peak
    return "unknown", None


def setup_training(config: TrainingConfig, device: torch.device):
    set_seed(config.seed)
    vocab = Vocabulary.from_config()
    model_cfg = load_model_config(PROJECT_ROOT / config.model_config)

    hands = load_hands(PROJECT_ROOT / config.data_dir)
    train_hands, _ = split_hands(hands, val_ratio=config.val_ratio)
    train_loader = make_dataloader(
        train_hands,
        vocab=vocab,
        batch_size=config.batch_size,
        block_size=config.block_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    model = PokerTransformer(model_cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    return vocab, model, optimizer, train_loader


def training_step(
    *,
    model: PokerTransformer,
    optimizer: AdamW,
    batch: Batch,
    vocab: Vocabulary,
    config: TrainingConfig,
    step: int,
) -> None:
    lr = learning_rate_at_step(step, config)
    set_optimizer_lr(optimizer, lr)

    losses = compute_losses(
        model,
        batch,
        pad_id=vocab.pad_id,
        value_loss_weight=config.value_loss_weight,
    )

    optimizer.zero_grad(set_to_none=True)
    losses.total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()


def move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        win_labels=batch.win_labels.to(device),
    )


def sum_profiler_flops(prof: profile) -> int:
    total = 0
    for event in prof.key_averages():
        if event.flops is not None and event.flops > 0:
            total += event.flops
    return total


def profile_training(
    config: TrainingConfig,
    *,
    warmup_steps: int,
    profile_steps: int,
    output_dir: Path,
    export_tensorboard: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available — profiler will record CPU activity only.")

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.json"

    vocab, model, optimizer, train_loader = setup_training(config, device)
    model.train()
    train_iter = iter(train_loader)

    def next_batch() -> Batch:
        nonlocal train_iter
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        return move_batch(batch, device)

    print(f"Warmup: {warmup_steps} steps (not profiled)...")
    for step in range(warmup_steps):
        training_step(
            model=model,
            optimizer=optimizer,
            batch=next_batch(),
            vocab=vocab,
            config=config,
            step=step,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    step_times: list[float] = []
    total_profile_wall = 0.0
    total_flops = 0

    print(f"Profiling: {profile_steps} steps...")
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        acc_events=True,
    ) as prof:
        for step in range(profile_steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            training_step(
                model=model,
                optimizer=optimizer,
                batch=next_batch(),
                vocab=vocab,
                config=config,
                step=warmup_steps + step,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            step_times.append(elapsed)
            total_profile_wall += elapsed
            prof.step()

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if export_tensorboard:
        tensorboard_trace_handler(str(output_dir))(prof)
        print(f"TensorBoard trace: {output_dir.resolve()}  →  tensorboard --logdir {output_dir}")
    else:
        prof.export_chrome_trace(str(trace_path))
    total_flops = sum_profiler_flops(prof)

    print("\n=== Profiler summary (top 15 by CUDA time) ===")
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(
        prof.key_averages().table(
            sort_by=sort_key,
            row_limit=15,
            top_level_events_only=True,
        )
    )

    avg_step = sum(step_times) / max(len(step_times), 1)
    print("\n=== Step timing ===")
    print(f"Profiled steps:     {profile_steps}")
    print(f"Avg wall time/step: {avg_step * 1000:.2f} ms")
    print(f"Total profile wall: {total_profile_wall:.2f} s")
    print(f"Throughput:         {profile_steps / max(total_profile_wall, 1e-9):.2f} steps/s")

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        gpu_key, peak_flops = detect_gpu_peak_flops(device_name)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)

        print("\n=== GPU memory ===")
        print(f"Device:             {device_name}")
        print(f"Matched peak table: {gpu_key}")
        print(f"Allocated:          {allocated / 1024**2:.1f} MiB")
        print(f"Reserved:           {reserved / 1024**2:.1f} MiB")
        print(f"Peak allocated:     {peak_allocated / 1024**2:.1f} MiB")

        print("\n=== FLOPs estimate ===")
        if total_flops > 0 and total_profile_wall > 0:
            achieved = total_flops / total_profile_wall
            print(f"Profiler FLOPs (sum): {total_flops:.3e}")
            print(f"Achieved FLOPs/s:     {achieved:.3e}")
            if peak_flops is not None:
                pct = 100.0 * achieved / peak_flops
                print(f"Peak FP16/BF16 ref:   {peak_flops:.3e} ({gpu_key})")
                print(f"% of advertised peak: {pct:.2f}%")
            else:
                print("Peak FP16/BF16 ref:   unknown GPU — add it to GPU_PEAK_FP16_FLOPS")
        else:
            print("FLOPs not reported by profiler for this run (with_flops may be unsupported).")
            print("Use the CUDA time table above for bottleneck analysis.")

    print("\n=== Trace files ===")
    if export_tensorboard:
        print(f"TensorBoard logdir: {output_dir.resolve()}")
    else:
        print(f"Chrome trace:  {trace_path.resolve()}")
        print("Open in chrome://tracing or https://ui.perfetto.dev/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a short training run.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "logs" / "profiler")
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Also write a TensorBoard-compatible trace (via tensorboard_trace_handler)",
    )
    args = parser.parse_args()

    config = load_training_config(args.config)
    profile_training(
        config,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        output_dir=args.output_dir,
        export_tensorboard=args.tensorboard,
    )


if __name__ == "__main__":
    main()
