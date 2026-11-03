#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENTS_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$EXPERIMENTS_DIR/base/common.sh"

EXPERIMENT=${1:?usage: run.sh <experiment-directory> [role overrides]}
shift
EXPERIMENT=$(resolve_experiment "$EXPERIMENT")

SERVER_ARGS=()
CLIENT_ARGS=()
SERVER_WORLD_SIZE_OVERRIDE=""
while (($#)); do
  case "$1" in
    --set)
      if (($# < 2)); then
        echo "Missing key=value after --set." >&2
        exit 2
      fi
      SERVER_ARGS+=(--set "$2")
      CLIENT_ARGS+=(--set "$2")
      shift 2
      ;;
    --server-set)
      if (($# < 2)); then
        echo "Missing key=value after --server-set." >&2
        exit 2
      fi
      SERVER_ARGS+=(--set "$2")
      if [[ $2 == server.world_size=* ]]; then
        SERVER_WORLD_SIZE_OVERRIDE=${2#*=}
      fi
      shift 2
      ;;
    --client-set)
      if (($# < 2)); then
        echo "Missing key=value after --client-set." >&2
        exit 2
      fi
      CLIENT_ARGS+=(--set "$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SERVER_WORLD_SIZE=${SERVER_WORLD_SIZE:-${SERVER_WORLD_SIZE_OVERRIDE:-1}}
if ((SERVER_WORLD_SIZE < 1)); then
  echo "SERVER_WORLD_SIZE must be positive." >&2
  exit 2
fi

PHYSICAL_GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
SERVER_VISIBLE_DEVICES=${SERVER_VISIBLE_DEVICES:-$(seq -s, 0 $((SERVER_WORLD_SIZE - 1)))}
CLIENT_VISIBLE_DEVICE=${CLIENT_VISIBLE_DEVICE:-$((PHYSICAL_GPU_COUNT - 1))}
IFS=',' read -r -a SERVER_GPU_LIST <<< "$SERVER_VISIBLE_DEVICES"
if ((${#SERVER_GPU_LIST[@]} != SERVER_WORLD_SIZE)); then
  echo "SERVER_VISIBLE_DEVICES must contain exactly SERVER_WORLD_SIZE GPU IDs." >&2
  exit 2
fi

declare -A USED_GPUS=()
for gpu in "${SERVER_GPU_LIST[@]}" "$CLIENT_VISIBLE_DEVICE"; do
  if [[ ! $gpu =~ ^[0-9]+$ ]] || ((gpu >= PHYSICAL_GPU_COUNT)); then
    echo "Invalid physical GPU $gpu; this host exposes $PHYSICAL_GPU_COUNT GPU(s)." >&2
    exit 2
  fi
  if [[ -n ${USED_GPUS[$gpu]:-} ]]; then
    echo "Client and server GPU assignments overlap at physical GPU $gpu." >&2
    exit 2
  fi
  USED_GPUS[$gpu]=1
done

RAY_PORT=${RAY_PORT:-26379}
export CUDA_VISIBLE_DEVICES="$SERVER_VISIBLE_DEVICES,$CLIENT_VISIBLE_DEVICE"
ray stop --force >/dev/null 2>&1 || true
ray start --head \
  --port="$RAY_PORT" \
  --num-gpus="$((SERVER_WORLD_SIZE + 1))" \
  --resources="{\"usl_server\": $SERVER_WORLD_SIZE, \"usl_client\": 1}" \
  --disable-usage-stats
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"

SERVER_PID=""
CLIENT_PID=""
cleanup() {
  if [[ -n $SERVER_PID ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n $CLIENT_PID ]] && kill -0 "$CLIENT_PID" 2>/dev/null; then
    kill "$CLIENT_PID" 2>/dev/null || true
  fi
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "Server physical GPUs: $SERVER_VISIBLE_DEVICES"
echo "Client physical GPU:   $CLIENT_VISIBLE_DEVICE"

"$EXPERIMENTS_DIR/base/server.sh" "$EXPERIMENT/server.yaml" \
  "${SERVER_ARGS[@]}" \
  --set run.mode=local \
  --set "server.world_size=$SERVER_WORLD_SIZE" &
SERVER_PID=$!

"$EXPERIMENTS_DIR/base/client.sh" "$EXPERIMENT/client.yaml" \
  "${CLIENT_ARGS[@]}" \
  --set run.mode=local &
CLIENT_PID=$!

if ! wait -n "$SERVER_PID" "$CLIENT_PID"; then
  exit 1
fi
wait "$SERVER_PID"
wait "$CLIENT_PID"
