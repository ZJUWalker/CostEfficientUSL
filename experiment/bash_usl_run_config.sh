#!/bin/bash

CONFIG_FILE="experiment/config.json"

# ================= 检查配置文件 =================
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found!"
    exit 1
fi

# ================= 获取实验总数 =================
# 利用 Python 读取 JSON 数组长度
NUM_CONFIGS=$(python -c "import json; print(len(json.load(open('$CONFIG_FILE'))))")

echo "Total experiments found: $NUM_CONFIGS"
echo "-------------------------------------"

# ================= 循环执行实验 =================
# Bash 的 C 语言风格循环，从 0 到 NUM_CONFIGS - 1
for ((i=0; i<NUM_CONFIGS; i++)); do
    
    echo ">> [ Experiment $((i+1)) / $NUM_CONFIGS ] Preparing..."

    # 1. 解析第 i 个配置
    # Python 脚本读取列表的第 i 项，并将 key 转大写，输出为 key="value" 格式
    eval $(python3 -c "import json; d=json.load(open('$CONFIG_FILE'))[$i]; print('\n'.join([f'{k.upper()}=\"{v}\"' for k,v in d.items()]))")

    # 2. 逻辑处理 (根据 PMODE 设置 MICRO_BATCH_SIZE)
    if [ "$PMODE" = "naive" ]; then
        MICRO_BATCH_SIZE=$BATCH_SIZE
    else
        MICRO_BATCH_SIZE=1
    fi

    echo "   Config: Model=$MODEL_NAME | Batch=$BATCH_SIZE | Split=$SPLIT_POINT | Offload(C/S)='$CLIENT_OFFLOAD'/'$SERVER_OFFLOAD'|Lora=$LORA|MBPS=$MBPS|WorldSize=$WORLD_SIZE"

    # 3. 启动 Server (后台)
    # 这里的变量已经通过 eval 在当前循环中更新了
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
        $PROFILE $LORA $SERVER_OFFLOAD &
    
    SERVER_PID=$!
    echo "   Server started (PID: $SERVER_PID). Waiting 10 seconds for server to start up..."
    sleep 10

    # 4. 启动 Client (前台)
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
        $LORA $CLIENT_OFFLOAD

    # 5. 等待本轮实验结束
    echo "   Client finished. Waiting for Server to exit..."
    wait $SERVER_PID
    
    echo ">> Experiment $((i+1)) Done."
    echo "-------------------------------------"
    
    # 稍微休息一下，确保端口释放，防止 Connection Refused
    sleep 2

done

echo "All experiments completed."