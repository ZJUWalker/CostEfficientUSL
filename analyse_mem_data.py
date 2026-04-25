import csv
import os
from collections import defaultdict


def load_data(filepath):
    """加载CSV数据并解析为字典列表"""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            for key in ['split_point', 'batch_size', 'offload_model_state_sp_num',
                        'client_offload_activation_mb_num', 'server_offload_activation_mb_num']:
                row[key] = int(row[key])
            for key in ['client_max_mem_alloc_mb', 'server_max_mem_alloc_mb']:
                row[key] = float(row[key])
            rows.append(row)
    return rows


def group_by(data, keys):
    """按多个key分组"""
    groups = defaultdict(list)
    for row in data:
        key = tuple(row[k] for k in keys)
        groups[key].append(row)
    return groups


def find_threshold_and_slope(points, var_key, mem_key, min_delta=1.0):
    """
    在一组按var_key排序的数据中，找到显存开始变化的有效阈值和之后的线性斜率。
    
    points: 同组数据点列表
    var_key: 变化变量名
    mem_key: 显存变量名 ('client_max_mem_alloc_mb' 或 'server_max_mem_alloc_mb')
    min_delta: 判定为"有效变化"的最小显存差值 (MB)
    
    返回: (threshold_val, slope, base_mem, is_valid, details)
        threshold_val: 有效阈值（显存开始变化的变量值），如果始终无变化则为None
        slope: 阈值后的每单位变化斜率 (MB/单位)
        base_mem: threshold_val对应的显存值
        is_valid: 是否检测到有效变化
        details: 详细分析字符串列表
    """
    if len(points) < 2:
        return None, 0, 0, False, ["数据点不足"]
    
    # 按变量值排序
    sorted_pts = sorted(points, key=lambda x: x[var_key])
    
    details = []
    details.append(f"  数据点: {[(p[var_key], p[mem_key]) for p in sorted_pts]}")
    
    # 从左到右扫描，找第一个显著变化
    threshold_idx = None
    threshold_val = None
    base_mem = None
    
    for i in range(1, len(sorted_pts)):
        var_prev = sorted_pts[i-1][var_key]
        var_curr = sorted_pts[i][var_key]
        mem_prev = sorted_pts[i-1][mem_key]
        mem_curr = sorted_pts[i][mem_key]
        delta = mem_curr - mem_prev
        var_step = var_curr - var_prev
        
        details.append(f"    {var_key} {var_prev}→{var_curr}: {mem_prev:.4f}→{mem_curr:.4f} "
                       f"(Δ={delta:+.4f}, step={var_step})")
        
        if abs(delta) >= min_delta:
            threshold_idx = i - 1  # 阈值之前的点是无效区最后一个点
            threshold_val = var_prev
            base_mem = mem_prev
            break
    
    if threshold_idx is None:
        # 全程无显著变化
        details.append(f"  ⚠ 全程无显著变化 (|Δ|<{min_delta})")
        return None, 0, sorted_pts[0][mem_key], False, details
    
    # 计算阈值之后的线性斜率
    # 从 threshold_idx+1 到末尾的所有点参与线性拟合
    linear_points = sorted_pts[threshold_idx:]
    
    if len(linear_points) < 2:
        details.append(f"  ⚠ 阈值后数据点不足，无法计算斜率")
        return threshold_val, 0, base_mem, True, details
    
    # 简单线性回归: slope = sum((x-x_mean)*(y-y_mean)) / sum((x-x_mean)^2)
    xs = [p[var_key] for p in linear_points]
    ys = [p[mem_key] for p in linear_points]
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    
    numerator = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    denominator = sum((xs[i] - x_mean) ** 2 for i in range(n))
    
    if abs(denominator) < 1e-9:
        slope = 0
    else:
        slope = numerator / denominator
    
    # 同时计算平均斜率（相邻点斜率的均值，更直观）
    avg_slopes = []
    for i in range(1, len(linear_points)):
        dv = linear_points[i][var_key] - linear_points[i-1][var_key]
        dm = linear_points[i][mem_key] - linear_points[i-1][mem_key]
        if dv != 0:
            avg_slopes.append(dm / dv)
    
    avg_slope = sum(avg_slopes) / len(avg_slopes) if avg_slopes else 0
    
    details.append(f"  ✓ 有效阈值: {var_key} >= {threshold_val} 时显存开始变化")
    details.append(f"  ✓ 阈值后线性斜率(回归): {slope:.4f} MB/单位")
    details.append(f"  ✓ 阈值后平均斜率: {avg_slope:.4f} MB/单位")
    
    return threshold_val, avg_slope, base_mem, True, details


def analyse_variable(data, var_key, control_keys, mem_keys, split_key, min_delta=1.0):
    """
    分析某个变量对显存的影响，自动检测阈值和线性段。
    
    var_key: 要分析的变量 ('offload_model_state_sp_num' 等)
    control_keys: 需要控制的变量列表
    mem_keys: 要分析的显存列表 [('client_max_mem_alloc_mb', '客户端'), ...]
    split_key: 分组键 ('split_point')
    """
    results = []
    all_details = []
    
    # 先按 split_point 分组
    split_groups = group_by(data, [split_key, 'batch_size'])
    
    for (sp, bs), sp_group in sorted(split_groups.items()):
        all_details.append(f"\n{'='*80}")
        all_details.append(f"【split_point={sp}, batch_size={bs}】分析 {var_key}")
        all_details.append(f"{'='*80}")
        
        # 在每组 split 内，按控制变量分组
        ctrl_groups = group_by(sp_group, control_keys)
        
        for ctrl_vals, group in sorted(ctrl_groups.items()):
            group = [g for g in group]  # copy
            if len(group) < 2:
                continue
            
            ctrl_desc = ", ".join(f"{k}={v}" for k, v in zip(control_keys, ctrl_vals))
            all_details.append(f"\n▶ 控制变量: {ctrl_desc}")
            
            for mem_key, mem_name in mem_keys:
                thresh, slope, base_mem, is_valid, details = find_threshold_and_slope(
                    group, var_key, mem_key, min_delta
                )
                all_details.extend(details)
                
                if is_valid:
                    results.append({
                        'split_point': sp,
                        'batch_size': bs,
                        'analysed_var': var_key,
                        'mem_type': mem_name,
                        'control_vars': ctrl_desc,
                        'threshold': thresh,
                        'slope': slope,
                        'base_mem_at_threshold': base_mem,
                    })
    
    return results, all_details


def print_summary(results, all_details, output_dir):
    """打印并保存汇总结果"""
    # 打印详细分析
    print("\n".join(all_details))
    
    # 按变量和 split_point 汇总
    print(f"\n{'='*80}")
    print("【汇总统计】")
    print(f"{'='*80}")
    
    # 分组统计 slope
    summary = defaultdict(list)
    for r in results:
        key = (r['split_point'], r['batch_size'], r['analysed_var'], r['mem_type'])
        summary[key].append(r['slope'])
    
    for key, slopes in sorted(summary.items()):
        sp, bs, var, mem = key
        n = len(slopes)
        mean = sum(slopes) / n
        variance = sum((s - mean) ** 2 for s in slopes) / n
        std = variance ** 0.5
        print(f"\n  split={sp}, bs={bs}, {var} → {mem}:")
        print(f"    样本数={n}, 平均斜率={mean:.4f}, 标准差={std:.4f}")
        print(f"    最小={min(slopes):.4f}, 最大={max(slopes):.4f}")
    
    # 输出建模公式
    print(f"\n{'='*80}")
    print("【建模公式参考】")
    print(f"{'='*80}")
    
    # 按 split_point 输出基准线
    split_groups = defaultdict(list)
    for r in results:
        split_groups[r['split_point']].append(r)
    
    for sp in sorted(split_groups.keys()):
        sp_results = split_groups[sp]
        print(f"\n  split_point={sp}:")
        
        # 提取各变量的典型阈值和斜率（取中位数或均值）
        for var in ['offload_model_state_sp_num', 'client_offload_activation_mb_num', 'server_offload_activation_mb_num']:
            for mem_type in ['客户端', '服务端']:
                sub = [r for r in sp_results if r['analysed_var'] == var and r['mem_type'] == mem_type]
                if not sub:
                    continue
                
                # 取阈值的中位数
                thresholds = [r['threshold'] for r in sub if r['threshold'] is not None]
                slopes = [r['slope'] for r in sub]
                
                if thresholds:
                    thresh = sorted(thresholds)[len(thresholds)//2]
                else:
                    thresh = 0
                
                if slopes:
                    slope = sum(slopes) / len(slopes)
                else:
                    slope = 0
                
                var_short = {'offload_model_state_sp_num': 'sp',
                             'client_offload_activation_mb_num': 'coa',
                             'server_offload_activation_mb_num': 'soa'}[var]
                
                print(f"    {mem_type} {var_short}: threshold>={thresh}, slope={slope:.4f} MB/单位")
    
    # 保存CSV
    if results:
        var_name = results[0]['analysed_var'].replace('offload_model_state_sp_num', 'sp')\
                                             .replace('client_offload_activation_mb_num', 'client_mb')\
                                             .replace('server_offload_activation_mb_num', 'server_mb')
        csv_path = os.path.join(output_dir, f"{var_name}_impact_analysis.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✓ 已保存: {csv_path}")


def build_cross_split_comparison(data):
    """比较不同 split_point 的基准显存差异"""
    print(f"\n{'='*80}")
    print("【split_point 对比分析】")
    print(f"{'='*80}")
    
    split_groups = group_by(data, ['split_point', 'batch_size'])
    
    for (sp, bs), group in sorted(split_groups.items()):
        # 找无卸载的基准
        baseline = [r for r in group 
                    if r['offload_model_state_sp_num'] == 0 
                    and r['client_offload_activation_mb_num'] == 0
                    and r['server_offload_activation_mb_num'] == 0]
        if baseline:
            b = baseline[0]
            print(f"\n  split_point={sp}, batch_size={bs}:")
            print(f"    基准显存 (无卸载): 客户端={b['client_max_mem_alloc_mb']:.4f} MB, 服务端={b['server_max_mem_alloc_mb']:.4f} MB")
        
        # 最大卸载时的显存
        max_unload = max(group, key=lambda r: (
            r['offload_model_state_sp_num'] + 
            r['client_offload_activation_mb_num'] + 
            r['server_offload_activation_mb_num']
        ))
        print(f"    最大卸载时: 客户端={max_unload['client_max_mem_alloc_mb']:.4f} MB, 服务端={max_unload['server_max_mem_alloc_mb']:.4f} MB")


def main():
    filepath = "log/analyse/qwen3-1.7b_mem_data.csv"
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    data = load_data(filepath)
    print(f"加载数据成功: {len(data)} 条记录")
    
    # 数据概览
    split_points = sorted(set(r['split_point'] for r in data))
    print(f"\nsplit_point 取值: {split_points}")
    for sp in split_points:
        sub = [r for r in data if r['split_point'] == sp]
        sp_nums = sorted(set(r['offload_model_state_sp_num'] for r in sub))
        coas = sorted(set(r['client_offload_activation_mb_num'] for r in sub))
        soas = sorted(set(r['server_offload_activation_mb_num'] for r in sub))
        print(f"  sp={sp}: sp_num={sp_nums}, coa={coas}, soa={soas}")
    
    output_dir = "log/analyse"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    all_details = []
    
    # 分析1: SP 对客户端显存的影响
    print(f"\n{'='*80}")
    print("【分析1】SP 卸载对显存的影响")
    print(f"{'='*80}")
    results, details = analyse_variable(
        data,
        var_key='offload_model_state_sp_num',
        control_keys=['client_offload_activation_mb_num', 'server_offload_activation_mb_num'],
        mem_keys=[('client_max_mem_alloc_mb', '客户端'), ('server_max_mem_alloc_mb', '服务端')],
        split_key='split_point',
        min_delta=1.0
    )
    print_summary(results, details, output_dir)
    all_results.extend(results)
    all_details.extend(details)
    
    # 分析2: client_mb 对显存的影响
    print(f"\n{'='*80}")
    print("【分析2】client_mb 卸载对显存的影响")
    print(f"{'='*80}")
    results, details = analyse_variable(
        data,
        var_key='client_offload_activation_mb_num',
        control_keys=['offload_model_state_sp_num', 'server_offload_activation_mb_num'],
        mem_keys=[('client_max_mem_alloc_mb', '客户端'), ('server_max_mem_alloc_mb', '服务端')],
        split_key='split_point',
        min_delta=1.0
    )
    print_summary(results, details, output_dir)
    all_results.extend(results)
    all_details.extend(details)
    
    # 分析3: server_mb 对显存的影响
    print(f"\n{'='*80}")
    print("【分析3】server_mb 卸载对显存的影响")
    print(f"{'='*80}")
    results, details = analyse_variable(
        data,
        var_key='server_offload_activation_mb_num',
        control_keys=['offload_model_state_sp_num', 'client_offload_activation_mb_num'],
        mem_keys=[('client_max_mem_alloc_mb', '客户端'), ('server_max_mem_alloc_mb', '服务端')],
        split_key='split_point',
        min_delta=1.0
    )
    print_summary(results, details, output_dir)
    all_results.extend(results)
    all_details.extend(details)
    
    # Split point 对比
    build_cross_split_comparison(data)
    
    # 保存完整报告
    report_path = os.path.join(output_dir, "analysis_report_v2.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"数据量: {len(data)} 条\n")
        f.write(f"split_points: {split_points}\n\n")
        f.write("\n".join(all_details))
    print(f"\n✓ 完整报告已保存: {report_path}")


if __name__ == "__main__":
    main()
