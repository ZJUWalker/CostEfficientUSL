from dataclasses import asdict, dataclass, field
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


@dataclass
class GanttChartData:
    mini_batch_idx: int = 0
    train_time_duration_ms: float = 0.0
    head_fwd_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    head_fwd_send_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    server_fwd_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    server_fwd_send_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    # tail_fwd_recv_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    tail_fwd_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    tail_bwd_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    tail_bwd_send_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    server_bwd_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    server_bwd_send_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    # head_bwd_recv_timestamp: List[float] = field(default_factory=lambda: [None] * 2)
    head_bwd_timestamp: List[float] = field(default_factory=lambda: [None] * 2)


def merge_cs_json(server_data: List[Dict], client_data: List[Dict], save_fp: str = 'merged.json', save: bool = False) -> List[Dict]:
    # 使用字典形式合并每个 mini_batch_idx 对应的数据

    for server_item, client_item in zip(server_data, client_data):
        # 通过 mini_batch_idx 进行合并
        if server_item['mini_batch_idx'] == client_item['mini_batch_idx']:
            # 合并：server 的非空数据覆盖 client 的空数据
            client_item["server_fwd_timestamp"] = server_item["server_fwd_timestamp"]
            client_item["server_bwd_timestamp"] = server_item["server_bwd_timestamp"]

        # 将合并后的数据写入 JSON 文件
    if save:
        with open(save_fp, 'w') as f:
            json.dump(client_data, f, indent=4)
    return client_data


def _to_aligned_ms(data_list: List[Dict]) -> List[Dict[str, List[Optional[int]]]]:
    """
    把一组 GanttChartData 转成毫秒整数，并以全局最小值为 0 对齐。

    Args:
        data_list: 多个 GanttChartData 对象

    Returns:
        List[Dict]，每个元素对应一个对齐后的 GanttChartData 的字段字典
    """
    # 收集所有有效时间戳
    all_vals = []
    for data in data_list:
        for field_name, value in data.items():
            if isinstance(value, list):
                for v in value:
                    if v is not None:
                        all_vals.append(v)

    if not all_vals:
        return []

    min_val = min(all_vals)

    aligned_list: List[Dict[str, List[Optional[int]]]] = []
    for data in data_list:
        aligned: Dict[str, List[Optional[int]]] = {}
        for field_name, value in data.items():
            if isinstance(value, list):
                new_list = []
                for v in value:
                    if v is None:
                        new_list.append(None)
                    else:
                        ms = int(round((v - min_val) * 1000, 2))
                        new_list.append(ms)
                aligned[field_name] = new_list
            else:
                aligned[field_name] = value
        aligned_list.append(aligned)

    return aligned_list


def save_gantt_chart_data(gantt_data_dict: Dict, fp: str):
    # 写入到JSON文件
    with open(fp, 'w') as f:
        json.dump(gantt_data_dict, f, indent=4)


# 阶段名字和颜色
STAGE_COLOR = {
    "head_fwd_timestamp": ("(C)Head Fwd", "#1f77b4"),  # 蓝色
    "head_fwd_send_timestamp": ("(C)Head Fwd Send", "#ff7f0e"),  # 橙色
    "server_fwd_timestamp": ("(S)Server Fwd", "#2ca02c"),  # 绿色
    "server_fwd_send_timestamp": ("(S)Server Fwd Send", "#d62728"),  # 红色
    "tail_fwd_timestamp": ("(C)Tail Fwd", "#9467bd"),  # 紫色
    "tail_bwd_timestamp": ("(C)Tail Bwd", "#8c564b"),  # 棕色
    "tail_bwd_send_timestamp": ("(C)Tail Bwd Send", "#e377c2"),  # 粉色
    "server_bwd_timestamp": ("(S)Server Bwd", "#7f7f7f"),  # 灰色
    "server_bwd_send_timestamp": ("(S)Server Bwd Send", "#bcbd22"),  # 黄绿色
    "head_bwd_timestamp": ("(C)Head Bwd", "#17becf"),  # 青色
}


def plot_gantt_per_batch(
    mini_batch_time_gantt: Optional[List[Dict] | List[GanttChartData]] = None,
    fp: str = "default.png",
    alpha: float = 0.3,
    show: bool = False,
):
    """
    把多个 GanttChartData 绘制成甘特图。
    每个 mini-batch 一行，不同阶段在同一行上用不同颜色表示。

    Args:
        mini_batch_time_gantt: GanttChartData 列表
        fp: 保存文件名
        alpha: 条形透明度，默认 0.6
        show: 是否直接 plt.show()
    """
    if not mini_batch_time_gantt:
        print("没有数据")
        return
    if isinstance(mini_batch_time_gantt[0], GanttChartData):
        mini_batch_time_gantt = [asdict(data) for data in mini_batch_time_gantt]
    # 一次性对齐所有 batch
    aligned_list = _to_aligned_ms(mini_batch_time_gantt)

    # 收集全局时间范围
    all_times = []
    for aligned in aligned_list:
        for key in STAGE_COLOR:
            interval = aligned.get(key)
            if interval and interval[0] is not None:
                all_times.extend(interval)
    if not all_times:
        print("没有有效的时间戳")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, aligned in enumerate(aligned_list):
        mb_idx = aligned["mini_batch_idx"]

        for key, (label, color) in STAGE_COLOR.items():
            interval = aligned.get(key)
            if not interval or interval[0] is None or interval[1] is None:
                continue

            start, end = interval
            duration = end - start
            ax.barh(
                y=idx,
                width=duration,
                left=start,
                height=0.5,
                color=color,
                edgecolor="black",
                alpha=alpha,
                label=label if idx == 0 else "",  # 避免重复图例
            )

    ax.set_xlabel("Time (ms, aligned)")
    ax.set_ylabel("Mini-batch")
    ax.set_yticks(range(len(aligned_list)))
    ax.set_yticklabels([f"MB{d['mini_batch_idx']}" for d in aligned_list])
    ax.set_title(f"Gantt Chart per Mini-Batch (One Row Each)(Config:{fp.split('/')[-1].split('.')[0]})")
    ax.grid(True, axis="x", linestyle="--", alpha=alpha)
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(fp, dpi=200)
        # print(f"Gantt 图已保存到 {fp}")


def plot_gantt_grouped(
    mini_batch_time_gantt: Optional[List[Dict] | List[GanttChartData]] = None,
    fp: str = "grouped.png",
    alpha: float = 0.3,
    show: bool = False,
):
    """
    绘制分组甘特图（四行：Server Send, Server Compute, Client Send, Client Compute）。

    Args:
        mini_batch_time_gantt: GanttChartData 列表或字典列表
        fp: 保存文件名
        alpha: 透明度
        show: 是否直接 plt.show()
    """
    if not mini_batch_time_gantt:
        print("没有数据")
        return
    if isinstance(mini_batch_time_gantt[0], GanttChartData):
        mini_batch_time_gantt = [asdict(data) for data in mini_batch_time_gantt]

    # 对齐时间
    aligned_list = _to_aligned_ms(mini_batch_time_gantt)

    # 收集全局时间范围
    all_times = []
    for aligned in aligned_list:
        for key in STAGE_COLOR:
            interval = aligned.get(key)
            if interval and interval[0] is not None:
                all_times.extend(interval)
    if not all_times:
        print("没有有效的时间戳")
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    # 四个分组的映射
    GROUP_MAPPING = {
        "Client Compute": ["head_fwd_timestamp", "head_bwd_timestamp", "tail_fwd_timestamp", "tail_bwd_timestamp"],
        "Client Send": ["head_fwd_send_timestamp", "tail_bwd_send_timestamp"],
        "Server Compute": ["server_fwd_timestamp", "server_bwd_timestamp"],
        "Server Send": ["server_fwd_send_timestamp", "server_bwd_send_timestamp"],
    }

    groups = list(GROUP_MAPPING.keys())

    for aligned in aligned_list:
        for row_idx, group_name in enumerate(groups):
            for key in GROUP_MAPPING[group_name]:
                interval = aligned.get(key)
                if not interval or interval[0] is None or interval[1] is None:
                    continue
                start, end = interval
                duration = end - start
                label, color = STAGE_COLOR[key]
                ax.barh(
                    y=row_idx,
                    width=duration,
                    left=start,
                    height=0.5,
                    color=color,
                    edgecolor="black",
                    alpha=alpha,
                    label=label if aligned["mini_batch_idx"] == 0 else "",  # 避免重复图例
                )

    ax.set_xlabel("Time (ms, aligned)")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_title(f"Grouped Gantt Chart (4 Rows)(Config:{fp.split('/')[-1].split('.')[0]})")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    # ax.legend()
    # ax.legend(
    #     loc="lower right",              # 把图例放在右下角
    #     bbox_to_anchor=(1.0, 0.05),     # 调整位置 (x=1.0 表示靠最右，y=0.05 表示靠下)
    #     fontsize=8,                     # 缩小字体
    #     frameon=True                    # 给图例加边框，避免和背景混
    # )
    ax.legend(
        fontsize=6,  # 再缩小字体
        markerscale=0.6,  # 再缩小 marker
        loc="lower right",  # 右下角
        bbox_to_anchor=(1, 0.2),  # 稍微上移，避免和 x 轴重叠
        frameon=True,
        borderaxespad=0.3,
        handlelength=1.0,
    )
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(fp, dpi=200)
