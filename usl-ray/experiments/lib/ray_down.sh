#!/usr/bin/env bash
set -euo pipefail

TOPOLOGY=${1:-same-node}
ray stop --force >/dev/null 2>&1 || true
if [[ "$TOPOLOGY" == "netns" ]]; then
  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  "$SCRIPT_DIR/tc/netns.sh" down
fi
