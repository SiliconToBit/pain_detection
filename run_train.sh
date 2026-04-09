#!/usr/bin/env bash
set -euo pipefail

# Defaults can be overridden by env vars.
IMAGE="${IMAGE:-pain_detection:latest}"
CONTAINER="${CONTAINER:-pain_detection}"
USE_GPU="${USE_GPU:-1}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -eq 0 ]; then
  CMD=(python3 code/train.py)
else
  CMD=("$@")
fi

GPU_ARGS=()
if [ "$USE_GPU" = "1" ]; then
  GPU_ARGS=(--gpus all)
fi

ENV_ARGS=()
FORWARD_VARS=(
  SAVE_DIR
  NUM_EPOCHS
  BATCH_SIZE
  LEARNING_RATE
  NUM_WORKERS
  MIN_EPOCHS
  EARLY_STOP_PATIENCE
  SCHEDULER_PATIENCE
  LOSS_TYPE
  MODEL_ARCH
  HIDDEN_DIM
  FEATURE_MODE
  USE_WEIGHTED_SAMPLER
  USE_FOCAL_LOSS
  FOCAL_GAMMA
  NORMALIZE_MODE
  CLASS_WEIGHTS
)

for var_name in "${FORWARD_VARS[@]}"; do
  if [ -n "${!var_name:-}" ]; then
    ENV_ARGS+=("-e" "$var_name=${!var_name}")
  fi
done

# Keep only one container with the fixed name.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

exec docker run \
  --name "$CONTAINER" \
  "${GPU_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  -v "$PROJECT_DIR":/pain_detection \
  -w /pain_detection \
  "$IMAGE" \
  "${CMD[@]}"
