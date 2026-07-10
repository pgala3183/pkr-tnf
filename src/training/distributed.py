"""PyTorch DistributedDataParallel (DDP) helpers for multi-GPU training."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class DistributedContext:
    """
    Bookkeeping for a torchrun-launched job.

    NCCL (NVIDIA Collective Communications Library) is the backend that moves
    gradient tensors between GPUs during DDP. After ``loss.backward()``, each
    replica holds **local** gradients computed on its own data shard. DDP registers
    hooks that fire during backward and perform an **all-reduce** on each
    gradient bucket: every rank contributes its partial sum, and every rank
    receives the **average** gradient (sum divided by world_size). That is why
    ``optimizer.step()`` can stay identical to single-GPU training — all replicas
    apply the same averaged update and stay in sync.
    """

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_main: bool


def is_dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> DistributedContext:
    """Initialize the process group (NCCL on CUDA) and bind this process to one GPU."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Distributed training requires torchrun environment variables "
            "(RANK, LOCAL_RANK, WORLD_SIZE). Launch via scripts/launch_distributed.sh."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        # NCCL knows how to all-reduce GPU tensors over NVLink/PCIe without
        # bouncing through host memory — that is why we prefer it over gloo here.
        backend = "nccl"
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        # CPU-only smoke tests fall back to gloo (no NCCL).
        backend = "glo"
        device = torch.device("cpu")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main=rank == 0,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(model: torch.nn.Module, ctx: DistributedContext) -> DDP | torch.nn.Module:
    if ctx.world_size == 1:
        return model
    if ctx.device.type == "cuda":
        return DDP(model, device_ids=[ctx.local_rank], output_device=ctx.local_rank)
    return DDP(model)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    # DDP (and some other wrappers) store the inner module at `.module`.
    if hasattr(model, "module"):
        return model.module  # type: ignore[return-value]
    return model


def barrier() -> None:
    if is_dist_available():
        dist.barrier()


@dataclass
class ThroughputStats:
    per_gpu_samples_per_sec: float
    aggregate_samples_per_sec: float
    scaling_efficiency: float | None


class ThroughputTracker:
    """Measure samples/sec over a logging window."""

    def __init__(self, batch_size: int, world_size: int) -> None:
        self.batch_size = batch_size
        self.world_size = world_size
        self._window_start = time.perf_counter()
        self._steps_in_window = 0
        self.single_gpu_baseline_sps: float | None = None

    def step(self) -> None:
        self._steps_in_window += 1

    def reset_window(self) -> None:
        self._window_start = time.perf_counter()
        self._steps_in_window = 0

    def measure_single_gpu_baseline(self, steps: int = 10) -> float:
        """Return samples/sec over the current window (rank-local)."""
        elapsed = max(time.perf_counter() - self._window_start, 1e-9)
        return (self._steps_in_window * self.batch_size) / elapsed

    def snapshot(
        self,
        *,
        baseline_single_gpu_sps: float | None = None,
    ) -> ThroughputStats | None:
        if self._steps_in_window == 0:
            return None

        elapsed = max(time.perf_counter() - self._window_start, 1e-9)
        local_samples = self._steps_in_window * self.batch_size
        per_gpu_sps = local_samples / elapsed

        if is_dist_available():
            # Gather per-rank throughputs so rank 0 can report aggregate fairly.
            local_tensor = torch.tensor([per_gpu_sps], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
            gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered, local_tensor)
            per_rank = [float(t.item()) for t in gathered]
            mean_per_gpu = sum(per_rank) / len(per_rank)
            aggregate_sps = mean_per_gpu * self.world_size
        else:
            mean_per_gpu = per_gpu_sps
            aggregate_sps = per_gpu_sps

        baseline = baseline_single_gpu_sps or self.single_gpu_baseline_sps
        scaling: float | None = None
        if baseline and baseline > 0:
            scaling = aggregate_sps / (self.world_size * baseline)

        return ThroughputStats(
            per_gpu_samples_per_sec=mean_per_gpu,
            aggregate_samples_per_sec=aggregate_sps,
            scaling_efficiency=scaling,
        )
