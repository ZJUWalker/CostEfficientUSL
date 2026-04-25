import os
import json
import csv
import glob

def extract_mem_data():
    # 目录路径
    dir_path = "log/profile/qwen/qwen3-1.7b"
    
    # 获取所有不包含 qloracomm/qlorcomm 的 json 文件
    all_json_files = glob.glob(os.path.join(dir_path, "*.json"))
    json_files = [f for f in all_json_files if "qloracomm" not in os.path.basename(f).lower() and "qlorcomm" not in os.path.basename(f).lower()]
    
    print(f"找到 {len(json_files)} 个不包含 'qlorcomm' 的 JSON 文件（共 {len(all_json_files)} 个）")
    
    # 需要提取的字段
    fields = [
        "split_point",
        "batch_size",
        "offload_model_state_sp_num",
        "client_offload_activation_mb_num",
        "server_offload_activation_mb_num",
        "client_max_mem_alloc_mb",
        "server_max_mem_alloc_mb"
    ]
    
    # 收集数据
    rows = []
    for filepath in sorted(json_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            row = {}
            for field in fields:
                row[field] = data.get(field, None)
            
            # 添加文件名便于追踪
            row["filename"] = os.path.basename(filepath)
            rows.append(row)
        except Exception as e:
            print(f"处理文件 {filepath} 时出错: {e}")
    
    # 写入 CSV
    output_path = "log/profile/qwen/qwen3-1.7b_mem_data.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        # 字段顺序：先放关键字段，再放文件名
        fieldnames = fields + ["filename"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n已成功提取 {len(rows)} 条记录到: {output_path}")
    
    # 打印前几条数据预览
    print("\n数据预览（前5行）:")
    print("-" * 120)
    header = " | ".join([f"{f:>30}" for f in fields])
    print(header)
    print("-" * 120)
    for row in rows[:5]:
        line = " | ".join([f"{str(row[f]):>30}" for f in fields])
        print(line)

if __name__ == "__main__":
    extract_mem_data()
