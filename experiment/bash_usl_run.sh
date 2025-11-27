SPLIT_POINT=2
MBPS=230
BATCH_SIZE=8
STEP=2
PMODE=gpipe
MODEL_NAME=qwen/qwen3-1.7b
PORT=9000
WORLD_SIZE=4
PROFILE='' #   '--prof' or '' 
OFFLOAD_ACTI='' # '--offload_activation' or ''

if [ "$PMODE" = "naive" ]; then
    MICRO_BATCH_SIZE=$BATCH_SIZE
else
    MICRO_BATCH_SIZE=1
fi

# 启动 server
python experiment/server_run_mp.py  --model=$MODEL_NAME --pmode=$PMODE --mbps=$MBPS \
    --batch_size=$BATCH_SIZE --micro_batch_size=$MICRO_BATCH_SIZE --split_point=$SPLIT_POINT \
    --step=$STEP --port=$PORT --world_size=$WORLD_SIZE $PROFILE $OFFLOAD_ACTI &
SERVER_PID=$!

# 启动 client
python experiment/client_run.py --model=$MODEL_NAME --pmode=$PMODE --mbps=$MBPS \
    --batch_size=$BATCH_SIZE --micro_batch_size=$MICRO_BATCH_SIZE --split_point=$SPLIT_POINT \
    --step=$STEP --port=$PORT --server_world_size=$WORLD_SIZE

# 等待 server 执行完成
wait $SERVER_PID