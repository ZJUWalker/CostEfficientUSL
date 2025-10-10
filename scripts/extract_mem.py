import json
import os
import pandas as pd

# 基础路径
base_path = 'log/profile/meta-llama/llama3.2-1b/'

# 输出文件路径
csv_output_path = 'log/extracted_mem_data_lora.csv'
split_points = [0, 1, 2, 3, 4]
mbps_values = [300]
offload_values = ['', '_oa', '_os', '_oa_os']
batch_sizes = [8]
all_data = []
keys = [
    "split_point",
    "batch_train_time_ms",
    "client_max_mem_alloc_mb",
    "server_max_mem_alloc_mb",
]
# 确保输出目录存在
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
# 遍历所有文件
for sp in split_points:
    for batch_size in batch_sizes:
        for mbps in mbps_values:
            for offload in offload_values:
                # 读取文件
                file_name = f"sp_{sp}_b_{batch_size}_mb_1_s_512_mbps_{mbps}_pipedream_wc_lora{offload}.json"
                file_path = os.path.join(base_path, file_name)
                with open(file_path, 'r') as f:
                    data: dict = json.load(f)
                # 解析数据
                # 提取需要的数据字段
                extracted_data = {key: data.get(key, None) for key in keys}
                extracted_data['offload_model_state'] = '√' if 'os' in offload else '×'
                extracted_data['offload_activation'] = '√' if 'oa' in offload else '×'
                print(extracted_data)
                # 写入csv文件
                # 将数据添加到所有数据列表
                all_data.append(extracted_data)

# 如果有提取的数据，保存为 CSV 文件
if all_data:
    df = pd.DataFrame(all_data)
    df = df.round(2)
    df.to_csv(csv_output_path, index=False)
    print(f"数据已成功保存为CSV文件: {csv_output_path}")
else:
    print("没有找到有效的数据文件")
