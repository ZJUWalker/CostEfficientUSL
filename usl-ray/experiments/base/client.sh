#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common.sh"

CONFIG=${1:?usage: client.sh <client.yaml> [--set key=value ...]}
shift
CONFIG=$(resolve_file "$CONFIG")
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray head address}"

PROFILE_ENABLED=$(bool_value "${SAVE_PROFILE:-1}")
GANTT_ENABLED=$(bool_value "${SAVE_GANTT:-$PROFILE_ENABLED}")
if [[ $PROFILE_ENABLED == false && $GANTT_ENABLED == true ]]; then
  echo "SAVE_GANTT requires SAVE_PROFILE." >&2
  exit 2
fi

cd "$PROJECT_DIR"
STARTED_NS=$("$PYTHON_BIN" -c "import time; print(time.time_ns())")
RUN_ARGS=("$@" --set "profile.enabled=$PROFILE_ENABLED")
DRY_RUN=false
for argument in "$@"; do
  if [[ $argument == "--dry-run" ]]; then
    DRY_RUN=true
  fi
done
usl_cli client --config "$CONFIG" "${RUN_ARGS[@]}"

if [[ $DRY_RUN == true || $PROFILE_ENABLED == false ]]; then
  exit 0
fi

PROFILE=$(
  "$PYTHON_BIN" - "$PROJECT_DIR" "$CONFIG" "$STARTED_NS" "${RUN_ARGS[@]}" <<'PY'
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

echo "Profile: $PROFILE"
if [[ $GANTT_ENABLED == true ]]; then
  GANTT=${GANTT_OUTPUT:-"$(dirname -- "$PROFILE")/gantt.png"}
  usl_cli plot "$PROFILE" --output "$GANTT"
  echo "Gantt:   $GANTT"
fi
