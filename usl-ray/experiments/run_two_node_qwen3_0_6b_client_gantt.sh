#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
CONFIG="$SCRIPT_DIR/configs/smoke-qwen3-0.6b/client.yaml"

SERVER_IP=${SERVER_IP:-10.124.0.24}
CLIENT_IP=${CLIENT_IP:-10.124.40.62}
RAY_PORT=${RAY_PORT:-26379}
CLIENT_GPU=${CLIENT_GPU:-0}

cd "$PROJECT_DIR"
ray stop --force >/dev/null 2>&1 || true

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

export CUDA_VISIBLE_DEVICES="$CLIENT_GPU"
ray start \
  --address="$SERVER_IP:$RAY_PORT" \
  --node-ip-address="$CLIENT_IP" \
  --num-gpus=1 \
  --resources='{"usl_client": 1}' \
  --disable-usage-stats

export RAY_ADDRESS="$SERVER_IP:$RAY_PORT"
STARTED_NS=$(python -c "import time; print(time.time_ns())")
RUN_ARGS=(--set profile.enabled=true "$@")

"$SCRIPT_DIR/run_client.sh" "$CONFIG" "${RUN_ARGS[@]}"

PROFILE=$(
  python - "$PROJECT_DIR" "$CONFIG" "$STARTED_NS" "${RUN_ARGS[@]}" <<'PY'
from pathlib import Path
import sys

from usl_ray.config import load_config

project_dir = Path(sys.argv[1])
config_path = Path(sys.argv[2])
started_ns = int(sys.argv[3])
arguments = sys.argv[4:]

overrides: list[str] = []
index = 0
while index < len(arguments):
    if arguments[index] != "--set" or index + 1 >= len(arguments):
        raise SystemExit(f"Expected --set key=value, got: {arguments[index:]}")
    overrides.append(arguments[index + 1])
    index += 2

config = load_config(config_path, overrides, role="client")
output_root = Path(config.run.output_dir)
if not output_root.is_absolute():
    output_root = project_dir / output_root
profiles = [
    path
    for path in output_root.rglob("profile.json")
    if path.stat().st_mtime_ns >= started_ns
]
if not profiles:
    raise SystemExit(f"No profile generated under {output_root}")
print(max(profiles, key=lambda path: path.stat().st_mtime_ns))
PY
)

GANTT=${GANTT_OUTPUT:-"$(dirname -- "$PROFILE")/gantt.png"}
usl-ray plot "$PROFILE" --output "$GANTT"

echo
echo "Profile: $PROFILE"
echo "Gantt:   $GANTT"
