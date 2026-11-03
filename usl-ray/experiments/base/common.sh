#!/usr/bin/env bash

BASE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENTS_DIR=$(cd -- "$BASE_DIR/.." && pwd)
PROJECT_DIR=$(cd -- "$EXPERIMENTS_DIR/.." && pwd)
PYTHON_BIN=${PYTHON:-python}
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

usl_cli() {
  if command -v usl-ray >/dev/null 2>&1; then
    usl-ray "$@"
  else
    "$PYTHON_BIN" -m usl_ray.cli "$@"
  fi
}

bool_value() {
  case "${1,,}" in
    1 | true | yes | on)
      echo true
      ;;
    0 | false | no | off)
      echo false
      ;;
    *)
      echo "Expected a boolean value, got: $1" >&2
      return 2
      ;;
  esac
}

resolve_experiment() {
  local experiment=$1
  if [[ $experiment != /* ]]; then
    if [[ -d $experiment ]]; then
      experiment=$(cd -- "$experiment" && pwd)
    elif [[ -d $PROJECT_DIR/$experiment ]]; then
      experiment=$(cd -- "$PROJECT_DIR/$experiment" && pwd)
    fi
  fi
  if [[ ! -f $experiment/client.yaml || ! -f $experiment/server.yaml ]]; then
    echo "Experiment must contain client.yaml and server.yaml: $experiment" >&2
    return 2
  fi
  echo "$experiment"
}

resolve_file() {
  local path=$1
  if [[ ! -f $path ]]; then
    echo "File does not exist: $path" >&2
    return 2
  fi
  local directory
  directory=$(cd -- "$(dirname -- "$path")" && pwd)
  echo "$directory/$(basename -- "$path")"
}

wait_for_tcp() {
  local host=$1
  local port=$2
  local timeout_s=$3
  "$PYTHON_BIN" - "$host" "$port" "$timeout_s" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.monotonic() + float(sys.argv[3])
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1):
            raise SystemExit(0)
    except OSError:
        time.sleep(1)
raise SystemExit(f"Timed out waiting for {host}:{port}")
PY
}
