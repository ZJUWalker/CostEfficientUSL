#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT=${1:?usage: run_single.sh <experiment-config-directory> [--set key=value ...]}
shift
SERVER_ARGS=()
CLIENT_ARGS=()
while (($#)); do
  case "$1" in
    --set)
      SERVER_ARGS+=(--set "$2")
      CLIENT_ARGS+=(--set "$2")
      shift 2
      ;;
    --server-set)
      SERVER_ARGS+=(--set "$2")
      shift 2
      ;;
    --client-set)
      CLIENT_ARGS+=(--set "$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
TOPOLOGY=${TOPO:-same-node}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_DIR"

source "$SCRIPT_DIR/lib/ray_up.sh" "$TOPOLOGY"
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
  source "$SCRIPT_DIR/lib/ray_down.sh"
}
trap cleanup EXIT INT TERM

usl-ray server --config "$EXPERIMENT/server.yaml" "${SERVER_ARGS[@]}" &
SERVER_PID=$!
${CLIENT_EXEC:-} usl-ray client --config "$EXPERIMENT/client.yaml" "${CLIENT_ARGS[@]}"
wait "$SERVER_PID"
