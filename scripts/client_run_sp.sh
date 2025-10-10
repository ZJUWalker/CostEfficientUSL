#!/bin/bash

# 参数范围
SP_LIST=(0)
MBPS_LIST=(300)
BATCH_SIZE=(8)


# 遍历组合
for SP in "${SP_LIST[@]}"; do
  for MBPS in "${MBPS_LIST[@]}"; do
    for BS in "${BATCH_SIZE[@]}"; do
      echo ">>> Running with SP=$SP, MBPS=$MBPS, batch_size=$BS"
      # 启动 server_run.sh，并传入参数（保证一一对应）
      bash scripts/server_run.sh $SP $MBPS &
      SERVER_PID=$!

      # 启动 client
      python experiment/client_run.py -SP=$SP --mbps=$MBPS --batch_size=$BS --micro_batch_size=1 --pmode=pipedream_wc --lora

      # 等待 server 执行完成
      wait $SERVER_PID
    done
  done
done
# do offload model
for SP in "${SP_LIST[@]}"; do
  for MBPS in "${MBPS_LIST[@]}"; do
    for BS in "${BATCH_SIZE[@]}"; do
      echo ">>> Running with SP=$SP, MBPS=$MBPS, batch_size=$BS, offload model" 
      # 启动 server_run.sh，并传入参数（保证一一对应）
      bash scripts/server_run.sh $SP $MBPS &
      SERVER_PID=$!

      # 启动 client
      python experiment/client_run.py -SP=$SP --mbps=$MBPS --batch_size=$BS --micro_batch_size=1 --pmode=pipedream_wc -OS --lora

      # 等待 server 执行完成
      wait $SERVER_PID
    done
  done
done
# do offload activation
for SP in "${SP_LIST[@]}"; do
  for MBPS in "${MBPS_LIST[@]}"; do
    for BS in "${BATCH_SIZE[@]}"; do
      echo ">>> Running with SP=$SP, MBPS=$MBPS, batch_size=$BS offload activation"
      # 启动 server_run.sh，并传入参数（保证一一对应）
      bash scripts/server_run.sh $SP $MBPS &
      SERVER_PID=$!

      # 启动 client
      python experiment/client_run.py -SP=$SP --mbps=$MBPS --batch_size=$BS --micro_batch_size=1 --pmode=pipedream_wc -OA --lora

      # 等待 server 执行完成
      wait $SERVER_PID
    done
  done
done
# do all offload
for SP in "${SP_LIST[@]}"; do
  for MBPS in "${MBPS_LIST[@]}"; do
    for BS in "${BATCH_SIZE[@]}"; do
      echo ">>> Running with SP=$SP, MBPS=$MBPS, batch_size=$BS offload all"
      # 启动 server_run.sh，并传入参数（保证一一对应）
      bash scripts/server_run.sh $SP $MBPS &
      SERVER_PID=$!

      # 启动 client
      python experiment/client_run.py -SP=$SP --mbps=$MBPS --batch_size=$BS --micro_batch_size=1 --pmode=pipedream_wc -OA -OS --lora

      # 等待 server 执行完成
      wait $SERVER_PID
    done
  done
done