#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
DEFAULT_EXPERIMENT="$SCRIPT_DIR/configs/smoke-qwen3-0.6b"

usage() {
  cat <<'EOF'
Usage:
  experiments/run_profile_gantt.sh [experiment-directory] [run_single overrides]

Examples:
  experiments/run_profile_gantt.sh
  experiments/run_profile_gantt.sh experiments/configs/qwen3-1.7b \
    --set train.steps=3 \
    --set profile.warmup_steps=1 \
    --server-set server.world_size=2

Environment:
  TOPO=same-node|netns       Ray topology (default: same-node)
  GANTT_OUTPUT=<path>        Override the generated PNG path

The default smoke experiment uses random model initialization and synthetic
data. same-node validates scheduling and profiling, but does not measure WAN
transfer time because Ray uses shared memory within one raylet.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

EXPERIMENT=$DEFAULT_EXPERIMENT
if (($#)) && [[ $1 != --* ]]; then
  EXPERIMENT=$1
  shift
fi
if [[ $EXPERIMENT != /* ]]; then
  if [[ -d $EXPERIMENT ]]; then
    EXPERIMENT=$(cd -- "$EXPERIMENT" && pwd)
  elif [[ -d $PROJECT_DIR/$EXPERIMENT ]]; then
    EXPERIMENT=$(cd -- "$PROJECT_DIR/$EXPERIMENT" && pwd)
  fi
fi
if [[ ! -f $EXPERIMENT/client.yaml || ! -f $EXPERIMENT/server.yaml ]]; then
  echo "Experiment must contain client.yaml and server.yaml: $EXPERIMENT" >&2
  exit 2
fi

for executable in ray usl-ray; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "Required command is unavailable: $executable" >&2
    echo "Install the project with: python -m pip install -e '$PROJECT_DIR[dev]'" >&2
    exit 127
  fi
done

cd "$PROJECT_DIR"
STARTED_NS=$(python -c "import time; print(time.time_ns())")
RUN_ARGS=(--set profile.enabled=true "$@")

echo "Running profiled experiment: $EXPERIMENT"
echo "Topology: ${TOPO:-same-node}"
"$SCRIPT_DIR/run_single.sh" "$EXPERIMENT" "${RUN_ARGS[@]}"

PROFILE=$(
  python - "$PROJECT_DIR" "$EXPERIMENT/client.yaml" "$STARTED_NS" "${RUN_ARGS[@]}" <<'PY'
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
    option = arguments[index]
    if option in {"--set", "--client-set"}:
        if index + 1 >= len(arguments):
            raise SystemExit(f"Missing value after {option}")
        overrides.append(arguments[index + 1])
        index += 2
    elif option == "--server-set":
        index += 2
    else:
        raise SystemExit(f"Unsupported run_single argument: {option}")

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
