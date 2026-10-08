#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:?usage: shape.sh apply|clear|show <interface> [peer-ip mbit delay-ms]}
INTERFACE=${2:?interface is required}

case "$ACTION" in
  apply)
    PEER=${3:?peer IP is required}
    MBIT=${4:?rate in Mbit/s is required}
    DELAY_MS=${5:?delay in milliseconds is required}
    tc qdisc replace dev "$INTERFACE" root handle 1: htb default 10
    tc class replace dev "$INTERFACE" parent 1: classid 1:10 htb rate 100gbit
    tc class replace dev "$INTERFACE" parent 1: classid 1:20 htb rate "${MBIT}mbit" ceil "${MBIT}mbit"
    tc qdisc replace dev "$INTERFACE" parent 1:20 handle 20: netem delay "${DELAY_MS}ms" limit 100000
    tc filter replace dev "$INTERFACE" protocol ip parent 1: prio 1 u32 \
      match ip dst "${PEER}/32" flowid 1:20
    ;;
  clear)
    tc qdisc del dev "$INTERFACE" root 2>/dev/null || true
    ;;
  show)
    tc -s qdisc show dev "$INTERFACE"
    tc -s class show dev "$INTERFACE"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
