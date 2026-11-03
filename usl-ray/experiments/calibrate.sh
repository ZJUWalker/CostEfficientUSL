#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT=${1:?usage: calibrate.sh <experiment-config-directory>}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_SP=${BASE_SP:-2}
BASE_BATCH=${BASE_BATCH:-8}
MAX_SPLIT_POINT=${MAX_SPLIT_POINT:?Set MAX_SPLIT_POINT to half the model layer count}
WORLD_SIZE=${WORLD_SIZE:-1}

COMBINATIONS=(
  "$BASE_SP $BASE_BATCH 0 0 0"
  "$BASE_SP $BASE_BATCH $BASE_BATCH 0 $BASE_BATCH"
  "$((BASE_SP + 1)) $BASE_BATCH 0 0 0"
  "$((BASE_SP + 1)) $BASE_BATCH $BASE_BATCH 0 $BASE_BATCH"
  "$BASE_SP $BASE_BATCH 0 $BASE_SP 0"
  "$((BASE_SP + 1)) $BASE_BATCH 0 $((BASE_SP + 1)) 0"
  "$BASE_SP $((BASE_BATCH + 1)) $((BASE_BATCH + 1)) 0 $((BASE_BATCH + 1))"
)

STARTED=$(date +%s)
for combination in "${COMBINATIONS[@]}"; do
  read -r split batch client_activation client_model server_activation <<<"$combination"
  "$SCRIPT_DIR/local/run.sh" "$EXPERIMENT" \
    --set "split.split_point=$split" \
    --set "batch.batch_size=$batch" \
    --server-set "server.world_size=$WORLD_SIZE" \
    --server-set "server.activation_offload_mb=$server_activation" \
    --client-set "client.activation_offload_mb=$client_activation" \
    --client-set "client.model_state_offload_layers=$client_model"
done

mapfile -t PROFILES < <(
  python - "$STARTED" <<'PY'
from pathlib import Path
import sys

started = int(sys.argv[1])
paths = [path for path in Path("runs").rglob("profile.json") if path.stat().st_mtime >= started]
for path in sorted(paths, key=lambda item: item.stat().st_mtime)[-7:]:
    print(path)
PY
)
if [[ ${#PROFILES[@]} -ne 7 ]]; then
  echo "Expected 7 calibration profiles, found ${#PROFILES[@]}" >&2
  exit 1
fi
usl-ray calibrate "${PROFILES[@]}" \
  --base-split-point "$BASE_SP" \
  --base-batch-size "$BASE_BATCH" \
  --max-split-point "$MAX_SPLIT_POINT" \
  --lora \
  --output runs/cost_model.json
