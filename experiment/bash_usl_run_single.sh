SPLIT_POINT=4
MBPS=230
BATCH_SIZE=8
STEP=2
PMODE=pdwc
MODEL_NAME=qwen/qwen3-8b
PORT=9000
WORLD_SIZE=1
PROFILE='' #   '--prof' or '' 
LORA='--lora' # '--lora' or ''
CLIENT_OFFLOAD='-OA -OS' # '--client_offload' or ''
SERVER_OFFLOAD='' # '--server_offload' or ''

if [ "$PMODE" = "naive" ]; then
    MICRO_BATCH_SIZE=$BATCH_SIZE
else
    MICRO_BATCH_SIZE=1
fi

# 启动 server
python experiment/server_run_mp.py  --model=$MODEL_NAME --pmode=$PMODE --mbps=$MBPS \
    --batch_size=$BATCH_SIZE --micro_batch_size=$MICRO_BATCH_SIZE --split_point=$SPLIT_POINT \
    --step=$STEP --port=$PORT --world_size=$WORLD_SIZE $PROFILE $LORA $SERVER_OFFLOAD &
SERVER_PID=$!
sleep 3
# 启动 client
python experiment/client_run.py --model=$MODEL_NAME --pmode=$PMODE --mbps=$MBPS \
    --batch_size=$BATCH_SIZE --micro_batch_size=$MICRO_BATCH_SIZE --split_point=$SPLIT_POINT \
    --step=$STEP --port=$PORT --server_world_size=$WORLD_SIZE $LORA $CLIENT_OFFLOAD

# 等待 server 执行完成
wait $SERVER_PID