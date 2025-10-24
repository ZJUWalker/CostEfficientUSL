#!/bin/bash

# 参数范围
MODEL_NAME=$1
SP=$2
BS=$3
MBPS=$4
COAM=$5
SOAM=$6
OSSP=$7
LORA=$8

echo ">>> Running with SP=$SP, MBPS=$MBPS, batch_size=$BS, offload client activation and model state, OSSP=$OSSP, COAM=$COAM, SOAM=$SOAM, LORA=$LORA <<<"
# 启动 server_run.sh，并传入参数（保证一一对应）
python experiment/server_run.py --model=$MODEL_NAME --pmode=pdwc --mbps=$MBPS --batch_size=$BS -SP=$SP -OAM=$SOAM  $LORA &
SERVER_PID=$!

# 启动 client
python experiment/client_run.py --model=$MODEL_NAME --pmode=pdwc  --mbps=$MBPS --batch_size=$BS -SP=$SP -OAM=$COAM -OSSP=$OSSP $LORA 

# 等待 server 执行完成
wait $SERVER_PID