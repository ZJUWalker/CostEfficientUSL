#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENTS_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

exec "$SCRIPT_DIR/run.sh" "$EXPERIMENTS_DIR/configs/smoke-qwen3-0.6b" "$@"
