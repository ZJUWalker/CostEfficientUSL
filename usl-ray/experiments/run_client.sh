#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: run_client.sh <client.yaml> [--set key=value ...]}
shift
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray head address}"
usl-ray client --config "$CONFIG" "$@"
