#!/usr/bin/env bash
set -euo pipefail

TOPOLOGY=${1:-same-node}
RAY_PORT=${RAY_PORT:-26379}
GPU_COUNT=${GPU_COUNT:-$(nvidia-smi -L | wc -l)}
USL_SERVER_SLOTS=${USL_SERVER_SLOTS:-$((GPU_COUNT - 1))}

if [[ "$TOPOLOGY" == "same-node" ]]; then
  ray start --head \
    --port="$RAY_PORT" \
    --num-gpus="$GPU_COUNT" \
    --resources="{\"usl_server\": $USL_SERVER_SLOTS, \"usl_client\": 1}" \
    --disable-usage-stats
  export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
elif [[ "$TOPOLOGY" == "netns" ]]; then
  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  NET_MBIT=${NET_MBIT:-230}
  NET_DELAY_MS=${NET_DELAY_MS:-10}
  USL_HOST_IP=${USL_HOST_IP:-10.200.0.1}
  USL_CLIENT_IP=${USL_CLIENT_IP:-10.200.0.2}
  "$SCRIPT_DIR/tc/netns.sh" up
  SERVER_GPUS=$(seq -s, 0 $((USL_SERVER_SLOTS - 1)))
  CUDA_VISIBLE_DEVICES="$SERVER_GPUS" ray start --head \
    --node-ip-address="$USL_HOST_IP" \
    --port="$RAY_PORT" \
    --num-gpus="$USL_SERVER_SLOTS" \
    --resources="{\"usl_server\": $USL_SERVER_SLOTS}" \
    --temp-dir=/tmp/usl-ray-server \
    --disable-usage-stats
  RAY_BIN=$(command -v ray)
  sudo ip netns exec usl_client sudo -u "$USER" env \
    "CUDA_VISIBLE_DEVICES=$USL_SERVER_SLOTS" \
    "$RAY_BIN" start \
    --address="$USL_HOST_IP:$RAY_PORT" \
    --node-ip-address="$USL_CLIENT_IP" \
    --num-gpus=1 \
    --resources='{"usl_client": 1}' \
    --disable-usage-stats
  sudo "$SCRIPT_DIR/tc/shape.sh" apply usl_srv0 "$USL_CLIENT_IP" "$NET_MBIT" "$NET_DELAY_MS"
  sudo ip netns exec usl_client "$SCRIPT_DIR/tc/shape.sh" apply usl_cli0 "$USL_HOST_IP" "$NET_MBIT" "$NET_DELAY_MS"
  export RAY_ADDRESS="$USL_HOST_IP:$RAY_PORT"
else
  echo "Unknown topology: $TOPOLOGY" >&2
  return 2
fi
export CLIENT_EXEC=""
