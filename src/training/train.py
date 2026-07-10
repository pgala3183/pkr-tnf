"""Train the poker action transformer on self-play token sequences."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DistributedSampler

from poker_transformer.model.transformer import ModelConfig, PokerTransformer, load_model_config
from poker_transformer.tokenizer.vocab import Vocabulary
from poker_transformer.training.data import (
    Batch,
    SelfPlayDataset,
    collate_hands,
    load_hands,
    make_dataloader,
    set_seed,
    split_hands,
)
from poker_transformer.training.distributed import (
    DistributedContext,
    ThroughputTracker,
    barrier,
    cleanup_distributed,
    setup_distributed,
    unwrap_model,
    wrap_ddp,
)
from poker_transformer.training.metrics import compute_losses, evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"
BASELINE_THROUGHPUT_PATH = PROJECT_ROOT / "logs" / "single_gpu_throughput.json"


@dataclass
class TrainingConfig:
    data_dir: str
    model_config: str
    batch_size: int
    num_workers: int
    block_size: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    warmup_steps: int
    max_steps: int
    gradient_clip: float
    value_loss_weight: float
    val_ratio: float
    eval_interval: int
    checkpoint_interval: int
    keep_last_checkpoints: int
    checkpoint_dir: str
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str | None
    log_csv: str
    log_interval: int
    seed: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainingConfig":
        betas = raw.get("betas", [0.9, 0.95])
        return cls(
            data_dir=str(raw["data_dir"]),
            model_config=str(raw["model_config"]),
            batch_size=int(raw["batch_size"]),
            num_workers=int(raw.get("num_workers", 0)),
            block_size=int(raw["block_size"]),
            learning_rate=float(raw["learning_rate"]),
            min_learning_rate=float(raw["min_learning_rate"]),
            weight_decay=float(raw["weight_decay"]),
            betas=(float(betas[0]), float(betas[1])),
            warmup_steps=int(raw["warmup_steps"]),
            max_steps=int(raw["max_steps"]),
            gradient_clip=float(raw.get("gradient_clip", 1.0)),
            value_loss_weight=float(raw["value_loss_weight"]),
            val_ratio=float(raw["val_ratio"]),
            eval_interval=int(raw["eval_interval"]),
            checkpoint_interval=int(raw["checkpoint_interval"]),
            keep_last_checkpoints=int(raw["keep_last_checkpoints"]),
            checkpoint_dir=str(raw["checkpoint_dir"]),
            use_wandb=bool(raw.get("use_wandb", False)),
            wandb_project=str(raw.get("wandb_project", "poker-transformer")),
            wandb_run_name=raw.get("wandb_run_name"),
            log_csv=str(raw["log_csv"]),
            log_interval=int(raw.get("log_interval", 10)),
            seed=int(raw.get("seed", 42)),
        )


def load_training_config(config_path: str | Path | None = None) -> TrainingConfig:
    path = Path(config_path) if config_path else DEFAULT_TRAINING_CONFIG
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TrainingConfig.from_dict(raw)


def learning_rate_at_step(step: int, config: TrainingConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    if step >= config.max_steps:
        return config.min_learning_rate

    progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


def set_optimizer_lr(optimizer: AdamW, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, keep_last: int) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = keep_last
        self.recent_paths: list[Path] = []
        self.best_val_loss = float("inf")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        path: Path,
        *,
        model: PokerTransformer,
        optimizer: AdamW,
        step: int,
        train_cfg: TrainingConfig,
        model_cfg: ModelConfig,
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        payload = {
            "step": step,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": train_cfg.__dict__,
            "model_config": model_cfg.__dict__,
            "val_metrics": val_metrics,
        }
        torch.save(payload, path)

        if path.name.startswith("step_"):
            self.recent_paths.append(path)
            while len(self.recent_paths) > self.keep_last:
                old = self.recent_paths.pop(0)
                if old.exists():
                    old.unlink()

    def maybe_save_best(
        self,
        *,
        model: PokerTransformer,
        optimizer: AdamW,
        step: int,
        train_cfg: TrainingConfig,
        model_cfg: ModelConfig,
        val_metrics: dict[str, float],
    ) -> bool:
        val_loss = val_metrics["total_loss"]
        if val_loss >= self.best_val_loss:
            return False

        self.best_val_loss = val_loss
        best_path = self.checkpoint_dir / "best.pt"
        self.save(
            best_path,
            model=model,
            optimizer=optimizer,
            step=step,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            val_metrics=val_metrics,
        )
        return True


class CSVLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [
            "timestamp",
            "step",
            "split",
            "total_loss",
            "action_loss",
            "value_loss",
            "action_perplexity",
            "learning_rate",
            "per_gpu_samples_per_sec",
            "aggregate_samples_per_sec",
            "scaling_efficiency",
        ]
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_split_metadata(
    path: Path,
    *,
    train_hand_ids: list[int],
    val_hand_ids: list[int],
    config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "val_ratio": config.val_ratio,
        "seed": config.seed,
        "train_hands": len(train_hand_ids),
        "val_hands": len(val_hand_ids),
        "train_hand_ids": train_hand_ids,
        "val_hand_ids": val_hand_ids,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def save_single_gpu_baseline(samples_per_sec: float) -> None:
    BASELINE_THROUGHPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples_per_sec": samples_per_sec,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with BASELINE_THROUGHPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_single_gpu_baseline() -> float | None:
    if not BASELINE_THROUGHPUT_PATH.exists():
        return None
    with BASELINE_THROUGHPUT_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload.get("samples_per_sec", 0)) or None


def _measure_local_throughput(
    model: PokerTransformer,
    optimizer: AdamW,
    batch: Batch,
    *,
    pad_id: int,
    value_loss_weight: float,
    gradient_clip: float,
    steps: int,
) -> float:
    """Run a short unwrapped loop to estimate single-GPU samples/sec."""
    model.train()
    start = time.perf_counter()
    for _ in range(steps):
        losses = compute_losses(
            model,
            batch,
            pad_id=pad_id,
            value_loss_weight=value_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    elapsed = max(time.perf_counter() - start, 1e-9)
    return (steps * batch.input_ids.size(0)) / elapsed


def train(
    config: TrainingConfig,
    *,
    distributed: bool = False,
    baseline_throughput: float | None = None,
) -> dict[str, Any]:
    ctx: DistributedContext | None = None
    if distributed:
        ctx = setup_distributed()
        device = ctx.device
        set_seed(config.seed + ctx.rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(config.seed)

    is_main = ctx.is_main if ctx else True
    world_size = ctx.world_size if ctx else 1

    vocab = Vocabulary.from_config()
    model_cfg = load_model_config(PROJECT_ROOT / config.model_config)
    if model_cfg.block_size != config.block_size:
        raise ValueError("block_size must match between training.yaml and model.yaml")

    hands = load_hands(PROJECT_ROOT / config.data_dir)
    train_hands, val_hands = split_hands(hands, val_ratio=config.val_ratio)
    if is_main:
        split_path = PROJECT_ROOT / config.data_dir / "split.json"
        save_split_metadata(
            split_path,
            train_hand_ids=[int(h["hand_id"]) for h in train_hands],
            val_hand_ids=[int(h["hand_id"]) for h in val_hands],
            config=config,
        )

    train_sampler: DistributedSampler | None = None
    if distributed and ctx is not None:
        train_sampler = DistributedSampler(
            SelfPlayDataset(train_hands),
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            seed=config.seed,
        )

    train_loader = make_dataloader(
        train_hands,
        vocab=vocab,
        batch_size=config.batch_size,
        block_size=config.block_size,
        shuffle=True,
        num_workers=config.num_workers,
        sampler=train_sampler,
    )
    val_loader = make_dataloader(
        val_hands,
        vocab=vocab,
        batch_size=config.batch_size,
        block_size=config.block_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = PokerTransformer(model_cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    # Measure rank-0 single-GPU throughput *before* DDP wrapping so we have a
    # fair baseline for scaling efficiency (no NCCL all-reduce in this phase).
    single_gpu_baseline = baseline_throughput or load_single_gpu_baseline()
    if distributed and ctx is not None and ctx.is_main and single_gpu_baseline is None:
        barrier()
        warmup_batch = next(iter(train_loader))
        warmup_batch = Batch(
            input_ids=warmup_batch.input_ids.to(device),
            attention_mask=warmup_batch.attention_mask.to(device),
            win_labels=warmup_batch.win_labels.to(device),
        )
        measured = _measure_local_throughput(
            model,
            optimizer,
            warmup_batch,
            pad_id=vocab.pad_id,
            value_loss_weight=config.value_loss_weight,
            gradient_clip=config.gradient_clip,
            steps=20,
        )
        single_gpu_baseline = measured
        save_single_gpu_baseline(measured)
        if is_main:
            print(f"Measured single-GPU baseline throughput: {measured:.1f} samples/s")
        barrier()
    elif distributed and ctx is not None:
        barrier()

    if distributed and ctx is not None:
        model = wrap_ddp(model, ctx)

    checkpoint_mgr = CheckpointManager(
        PROJECT_ROOT / config.checkpoint_dir,
        keep_last=config.keep_last_checkpoints,
    )
    csv_logger = CSVLogger(PROJECT_ROOT / config.log_csv) if is_main else None

    wandb_run = None
    if config.use_wandb and is_main:
        import wandb

        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={**config.__dict__, **model_cfg.__dict__, "distributed": distributed, "world_size": world_size},
        )

    throughput = ThroughputTracker(config.batch_size, world_size)
    throughput.single_gpu_baseline_sps = single_gpu_baseline

    step = 0
    sampler_epoch = 0
    train_iter = iter(train_loader)
    last_val_metrics: dict[str, float] | None = None

    if is_main:
        mode = f"DDP x{world_size}" if distributed else "single-GPU"
        print(f"Training on {device} ({mode}), batch_size={config.batch_size} per GPU")

    while step < config.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            sampler_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(sampler_epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        lr = learning_rate_at_step(step, config)
        set_optimizer_lr(optimizer, lr)

        batch = Batch(
            input_ids=batch.input_ids.to(device, non_blocking=True),
            attention_mask=batch.attention_mask.to(device, non_blocking=True),
            win_labels=batch.win_labels.to(device, non_blocking=True),
        )

        train_losses = compute_losses(
            model,
            batch,
            pad_id=vocab.pad_id,
            value_loss_weight=config.value_loss_weight,
        )

        # DDP hooks all-reduce averaged gradients during this backward pass.
        optimizer.zero_grad(set_to_none=True)
        train_losses.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        throughput.step()

        if step % config.log_interval == 0 and is_main:
            timestamp = datetime.now(timezone.utc).isoformat()
            tp = throughput.snapshot(baseline_single_gpu_sps=single_gpu_baseline)
            row = {
                "timestamp": timestamp,
                "step": step,
                "split": "train",
                "total_loss": train_losses.total_loss.item(),
                "action_loss": train_losses.action_loss.item(),
                "value_loss": train_losses.value_loss.item(),
                "action_perplexity": train_losses.action_perplexity,
                "learning_rate": lr,
                "per_gpu_samples_per_sec": tp.per_gpu_samples_per_sec if tp else "",
                "aggregate_samples_per_sec": tp.aggregate_samples_per_sec if tp else "",
                "scaling_efficiency": tp.scaling_efficiency if tp and tp.scaling_efficiency is not None else "",
            }
            if csv_logger is not None:
                csv_logger.log(row)
            msg = (
                f"step {step:5d} | train loss {train_losses.total_loss.item():.4f} "
                f"(action {train_losses.action_loss.item():.4f}, "
                f"value {train_losses.value_loss.item():.4f}) | lr {lr:.2e}"
            )
            if tp is not None:
                msg += (
                    f" | per-GPU {tp.per_gpu_samples_per_sec:.1f} samples/s"
                    f" | aggregate {tp.aggregate_samples_per_sec:.1f} samples/s"
                )
                if tp.scaling_efficiency is not None:
                    msg += f" | scaling eff {tp.scaling_efficiency:.2f}"
            print(msg)
            throughput.reset_window()
            if wandb_run is not None and tp is not None:
                wandb_run.log(
                    {
                        "train/total_loss": train_losses.total_loss.item(),
                        "train/per_gpu_samples_per_sec": tp.per_gpu_samples_per_sec,
                        "train/aggregate_samples_per_sec": tp.aggregate_samples_per_sec,
                        "train/scaling_efficiency": tp.scaling_efficiency,
                    },
                    step=step,
                )

        if step > 0 and step % config.eval_interval == 0 and is_main:
            last_val_metrics = evaluate(
                unwrap_model(model),
                val_loader,
                pad_id=vocab.pad_id,
                value_loss_weight=config.value_loss_weight,
                device=device,
            )
            if csv_logger is not None:
                csv_logger.log(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "step": step,
                        "split": "val",
                        "total_loss": last_val_metrics["total_loss"],
                        "action_loss": last_val_metrics["action_loss"],
                        "value_loss": last_val_metrics["value_loss"],
                        "action_perplexity": last_val_metrics["action_perplexity"],
                        "learning_rate": lr,
                        "per_gpu_samples_per_sec": "",
                        "aggregate_samples_per_sec": "",
                        "scaling_efficiency": "",
                    }
                )
            print(
                f"step {step:5d} | val loss {last_val_metrics['total_loss']:.4f} "
                f"(action {last_val_metrics['action_loss']:.4f}, "
                f"ppl {last_val_metrics['action_perplexity']:.2f})"
            )
            if wandb_run is not None:
                wandb_run.log({f"val/{k}": v for k, v in last_val_metrics.items()}, step=step)

            improved = checkpoint_mgr.maybe_save_best(
                model=model,
                optimizer=optimizer,
                step=step,
                train_cfg=config,
                model_cfg=model_cfg,
                val_metrics=last_val_metrics,
            )
            if improved:
                print(f"step {step:5d} | saved new best checkpoint (val loss {last_val_metrics['total_loss']:.4f})")
            model.train()

        if step > 0 and step % config.checkpoint_interval == 0 and is_main:
            ckpt_path = checkpoint_mgr.checkpoint_dir / f"step_{step:06d}.pt"
            checkpoint_mgr.save(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                step=step,
                train_cfg=config,
                model_cfg=model_cfg,
                val_metrics=last_val_metrics,
            )
            print(f"step {step:5d} | saved checkpoint {ckpt_path.name}")

        step += 1

    if is_main and last_val_metrics is None:
        last_val_metrics = evaluate(
            unwrap_model(model),
            val_loader,
            pad_id=vocab.pad_id,
            value_loss_weight=config.value_loss_weight,
            device=device,
        )

    final_tp = throughput.snapshot(baseline_single_gpu_sps=single_gpu_baseline) if is_main else None
    if is_main and final_tp is not None:
        print(
            f"\nThroughput: per-GPU {final_tp.per_gpu_samples_per_sec:.1f} samples/s | "
            f"aggregate {final_tp.aggregate_samples_per_sec:.1f} samples/s"
        )
        if final_tp.scaling_efficiency is not None:
            print(
                f"Scaling efficiency: {final_tp.scaling_efficiency:.2f} "
                f"(aggregate / ({world_size} × {single_gpu_baseline:.1f} single-GPU baseline))"
            )
            if not distributed:
                save_single_gpu_baseline(final_tp.per_gpu_samples_per_sec)

    if wandb_run is not None:
        wandb_run.finish()

    if distributed:
        cleanup_distributed()

    summary = {
        "steps": step,
        "train_hands": len(train_hands),
        "val_hands": len(val_hands),
        "final_val_metrics": last_val_metrics,
        "best_val_loss": checkpoint_mgr.best_val_loss if is_main else None,
        "distributed": distributed,
        "world_size": world_size,
    }
    if final_tp is not None:
        summary["per_gpu_samples_per_sec"] = final_tp.per_gpu_samples_per_sec
        summary["aggregate_samples_per_sec"] = final_tp.aggregate_samples_per_sec
        summary["scaling_efficiency"] = final_tp.scaling_efficiency
        summary["single_gpu_baseline_sps"] = single_gpu_baseline

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train poker-transformer on self-play data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use PyTorch DDP (launch with torchrun / scripts/launch_distributed.sh)",
    )
    parser.add_argument(
        "--baseline-throughput",
        type=float,
        default=None,
        help="Single-GPU samples/sec baseline for scaling efficiency (optional)",
    )
    args = parser.parse_args()

    config = load_training_config(args.config)
    summary = train(
        config,
        distributed=args.distributed,
        baseline_throughput=args.baseline_throughput,
    )
    if summary.get("best_val_loss") is not None or not args.distributed:
        print("\n=== Training complete ===")
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
