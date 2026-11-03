#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
CONFIG="$SCRIPT_DIR/configs/smoke-qwen3-0.6b/server.yaml"

SERVER_IP=${SERVER_IP:-10.124.0.24}
RAY_PORT=${RAY_PORT:-26379}
SERVER_GPU=${SERVER_GPU:-0}

cd "$PROJECT_DIR"
ray stop --force >/dev/null 2>&1 || true

export CUDA_VISIBLE_DEVICES="$SERVER_GPU"
ray start --head \
  --node-ip-address="$SERVER_IP" \
  --port="$RAY_PORT" \
  --num-gpus=1 \
  --resources='{"usl_server": 1}' \
  --disable-usage-stats

export RAY_ADDRESS="$SERVER_IP:$RAY_PORT"
echo "Ray server ready at $RAY_ADDRESS; waiting for the client."

if ! "$SCRIPT_DIR/run_server.sh" "$CONFIG" --set profile.enabled=true "$@"; then
  ray stop --force >/dev/null 2>&1 || true
  exit 1
fi

echo "Server run completed. Keep Ray running until the client has written its profile."
echo "Cleanup command: ray stop --force"
