#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:?usage: netns.sh up|down|show}
NAMESPACE=${USL_NETNS:-usl_client}
HOST_IF=${USL_HOST_IF:-usl_srv0}
CLIENT_IF=${USL_CLIENT_IF:-usl_cli0}
HOST_IP=${USL_HOST_IP:-10.200.0.1}
CLIENT_IP=${USL_CLIENT_IP:-10.200.0.2}

down() {
  sudo ip netns del "$NAMESPACE" 2>/dev/null || true
  sudo ip link del "$HOST_IF" 2>/dev/null || true
}

case "$ACTION" in
  up)
    down
    sudo ip netns add "$NAMESPACE"
    sudo ip link add "$HOST_IF" type veth peer name "$CLIENT_IF"
    sudo ip link set "$CLIENT_IF" netns "$NAMESPACE"
    sudo ip addr add "$HOST_IP/30" dev "$HOST_IF"
    sudo ip link set "$HOST_IF" up
    sudo ip netns exec "$NAMESPACE" ip addr add "$CLIENT_IP/30" dev "$CLIENT_IF"
    sudo ip netns exec "$NAMESPACE" ip link set lo up
    sudo ip netns exec "$NAMESPACE" ip link set "$CLIENT_IF" up
    ;;
  down)
    down
    ;;
  show)
    ip -br addr show "$HOST_IF"
    sudo ip netns exec "$NAMESPACE" ip -br addr show "$CLIENT_IF"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
