#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENTS_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
EXPERIMENT=${1:-"$EXPERIMENTS_DIR/configs/smoke-qwen3-0.6b"}
COST_MODEL=${2:-${COST_MODEL:-"$EXPERIMENTS_DIR/../runs/cost_model.json"}}

read -r -a SPLITS <<< "${SPLIT_POINTS:-4}"
read -r -a BATCHES <<< "${BATCH_SIZES:-8}"

for split in "${SPLITS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    "$SCRIPT_DIR/run.sh" "$EXPERIMENT" "$COST_MODEL" \
      --set "split.split_point=$split" \
      --set "batch.batch_size=$batch"
  done
done
