#!/usr/bin/env bash
# Launch multi-GPU DDP training with torchrun.
#
# Usage:
#   ./scripts/launch_distributed.sh 4
#   ./scripts/launch_distributed.sh 4 configs/training_smoke.yaml
#
# Replace 4 with the number of GPUs on the machine (N).

set -euo pipefail

NPROC="${1:?Usage: launch_distributed.sh <num_gpus> [training_config]}"
CONFIG="${2:-configs/training.yaml}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT"

echo "Launching DDP training on ${NPROC} GPU(s) with config ${CONFIG}"

torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  -m poker_transformer.training.train \
  --distributed \
  --config "${CONFIG}"
