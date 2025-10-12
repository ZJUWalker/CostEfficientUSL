#!/bin/bash
# server_run_2.sh
# 用于被 client_run_1.sh 调用
# 参数: $1 -> SP, $2 -> mbps

SP=$1
mbps=$2

echo "[Server] Starting server_run.py for SP=${SP}, mbps=${mbps}"
python experiment/server_run.py -SP=${SP} --mbps=${mbps} --pmode=pdwc
echo "[Server] Finished server_run.py for SP=${SP}, mbps=${mbps}"
