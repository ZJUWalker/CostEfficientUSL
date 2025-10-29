import json
import os
import pandas as pd

# 基础路径
model = 'qwen/qwen3-4b'
mbps = 230
lora = True
base_path = f'log/profile/{model}'


split_points = list(range(1, 18))
batch_sizes = [8, 16, 24, 32]
# oam = [0, 4, 8]
# osr = [0, 0.5, 1.0]
all_data = []
# 遍历所有文件
for sp in split_points:
    osr = [0, sp]
    for batch_size in batch_sizes:
        # if (sp == 8 and batch_size == 8) or (sp == 4 and batch_size == 16):
        #     continue
        oam = [0, batch_size]
        for oa in oam:
            for osr_ in osr:
                # 读取文件
                # if oa == 0:
                #     file_name = f"sp_{sp}_b_{batch_size}_mb_1_s_512_mbps_{mbps}_pipedream_wc.json"
                # else:
                file_name = f"sp_{sp}_b_{batch_size}_mb_1_s_512_mbps_{mbps}_pipedream_wc{f'_lora' if lora else ''}{f'_coa_{oa}' if oa > 0 else ''}{f'_cos_{osr_}' if osr_ > 0 else ''}{f'_soa_{oa}' if oa > 0 else ''}.json"
                file_path = os.path.join(base_path, file_name)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                except FileNotFoundError:
                    print(f"File {file_path} not found.")
                    continue
                # 解析数据
                # 提取需要的数据字段
                extracted_data = {
                    "batch_size": data.get("batch_size", None),
                    "split_point": sp,
                    "offload_mb_num": oa,
                    "offload_ms_sp_num": osr_,
                    # "mbps": mbps,
                    "client_mem": data.get("client_max_mem_alloc_mb", None),
                    "server_mem": data.get("server_max_mem_alloc_mb", None),
                    "batch_time": data.get("batch_train_time_ms", None),
                    # "offload_mem_alloc_mb": oa,
                }

                # 写入csv文件
                # 将数据添加到所有数据列表
                all_data.append(extracted_data)

# 写入csv文件
df = pd.DataFrame(all_data)
df = df.round(2)

# 输出文件路径
csv_output_path = f'log/extracted_optim/{model.split('/')[1]}{'_lora' if lora else ''}.csv'
# 确保输出目录存在
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
df.to_csv(csv_output_path, index=False)
print(f"数据已成功保存为CSV文件: {csv_output_path}")
