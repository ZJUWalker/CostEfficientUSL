#!/bin/bash

# 参数范围
SP_LIST=(1 2 3 4)
MBPS_LIST=(100 500 1000 2000)
MICRO_BATCH_SIZE_LIST=(1 4)

# 遍历组合
for SP in "${SP_LIST[@]}"; do
  for MBPS in "${MBPS_LIST[@]}"; do
    for MBS in "${MICRO_BATCH_SIZE_LIST[@]}"; do
      echo ">>> Running with SP=$SP, MBPS=$MBPS, micro_batch_size=$MBS"
      # 启动 server_run.sh，并传入参数（保证一一对应）
      bash scripts/server_run.sh $SP $MBPS &
      SERVER_PID=$!

      # 启动 client
      python experiment/client_run.py -SP=$SP --mbps=$MBPS --micro_batch_size=$MBS --async_io

      # 等待 server 执行完成
      wait $SERVER_PID
    done
  done
done