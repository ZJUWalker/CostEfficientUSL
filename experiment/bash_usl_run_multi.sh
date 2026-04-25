#!/bin/bash

# ================= 配置区域 (枚举变量) =================
# 在括号内添加更多值以进行网格搜索，用空格分隔
# 例如: SPLIT_POINTS=(4 8 16)

SPLIT_POINTS=(3)
BATCH_SIZES=(8)
MODEL_NAMES=("qwen/qwen3-1.7b")
WORLD_SIZES=(1)

# Offload 层数配置 (0 表示不卸载该类型)
# -OAM: 激活量卸载的层数
# -OSSP: 模型卸载的层数
CLIENT_OAM_LAYERS=(0 1 2 4 6 8)
CLIENT_OSSP_LAYERS=(0 1 2 3)
SERVER_OAM_LAYERS=(0 1 2 4 6 8)
SERVER_OSSP_LAYERS=(0)

# ================= 固定参数 =================
MBPS=230
STEP=3
PMODE=pdwc
PORT=9000 # 初始端口，如果并行跑可能需要动态调整，但在串行循环中固定即可
PROFILE='--prof' # '--prof' or ''
LORA='--lora'
QUANTIZATION=''

# ================= 主循环逻辑 =================

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for SPLIT_POINT in "${SPLIT_POINTS[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for CLIENT_OAM in "${CLIENT_OAM_LAYERS[@]}"; do
          for CLIENT_OSSP in "${CLIENT_OSSP_LAYERS[@]}"; do
            for SERVER_OAM in "${SERVER_OAM_LAYERS[@]}"; do
              for SERVER_OSSP in "${SERVER_OSSP_LAYERS[@]}"; do
                for WORLD_SIZE in "${WORLD_SIZES[@]}"; do
                
                # 构建 Client offload 参数
                CLIENT_OFFLOAD=""
                if [ "$CLIENT_OAM" -ne 0 ]; then
                  CLIENT_OFFLOAD="$CLIENT_OFFLOAD -OAM=$CLIENT_OAM"
                fi
                if [ "$CLIENT_OSSP" -ne 0 ]; then
                  CLIENT_OFFLOAD="$CLIENT_OFFLOAD -OSSP=$CLIENT_OSSP"
                fi
                CLIENT_OFFLOAD=$(echo "$CLIENT_OFFLOAD" | sed 's/^ *//')

                # 构建 Server offload 参数
                SERVER_OFFLOAD=""
                if [ "$SERVER_OAM" -ne 0 ]; then
                  SERVER_OFFLOAD="$SERVER_OFFLOAD -OAM=$SERVER_OAM"
                fi
                if [ "$SERVER_OSSP" -ne 0 ]; then
                  SERVER_OFFLOAD="$SERVER_OFFLOAD -OSSP=$SERVER_OSSP"
                fi
                SERVER_OFFLOAD=$(echo "$SERVER_OFFLOAD" | sed 's/^ *//')
                
                echo "========================================================================"
                echo "正在运行实验配置:"
                echo "Model: $MODEL_NAME | World Size: $WORLD_SIZE | Split: $SPLIT_POINT"
                echo "Batch: $BATCH_SIZE | Client Offload: '$CLIENT_OFFLOAD' | Server Offload: '$SERVER_OFFLOAD'"
                echo "========================================================================"

                # 根据 PMODE 设置 MICRO_BATCH_SIZE
                if [ "$PMODE" = "naive" ]; then
                    MICRO_BATCH_SIZE=$BATCH_SIZE
                else
                    MICRO_BATCH_SIZE=1
                fi

                # 启动 Server (后台运行)
                # 注意：$SERVER_OFFLOAD 变量不加引号，以便让 shell 正确解析为空或参数
                python experiment/server_run_mp.py \
                    --model="$MODEL_NAME" \
                    --pmode="$PMODE" \
                    --mbps=$MBPS \
                    --batch_size=$BATCH_SIZE \
                    --micro_batch_size=$MICRO_BATCH_SIZE \
                    --split_point=$SPLIT_POINT \
                    --step=$STEP \
                    --port=$PORT \
                    --world_size=$WORLD_SIZE \
                    $PROFILE $LORA $SERVER_OFFLOAD $QUANTIZATION &
                
                SERVER_PID=$!
                echo "Server PID: $SERVER_PID started. Waiting 3s..."
                sleep 5

                # 启动 Client (前台运行)
                python experiment/client_run.py \
                    --model="$MODEL_NAME" \
                    --pmode="$PMODE" \
                    --mbps=$MBPS \
                    --batch_size=$BATCH_SIZE \
                    --micro_batch_size=$MICRO_BATCH_SIZE \
                    --split_point=$SPLIT_POINT \
                    --step=$STEP \
                    --port=$PORT \
                    --server_world_size=$WORLD_SIZE \
                    $LORA $CLIENT_OFFLOAD $QUANTIZATION

                # 实验结束后的清理工作
                echo "Client finished. Waiting for Server to exit..."
                
                # 等待 server 进程结束 (如果 client 跑完 server 也会自动退出的话)
                # 为了防止 server 没退出导致端口占用，这里做一个超时检测或者强制 kill 也是常见的保险做法
                wait $SERVER_PID
                
                # 这里的 sleep 是为了确保端口被系统完全释放
                sleep 2
                echo "Experiment finished."
                echo ""

                done
              done
            done
          done
        done
      done
    done
  done
done

echo "所有实验组已完成。"
