from dataclasses import asdict, dataclass, field
import json
import matplotlib.pyplot as plt
from typing import Any, List, Dict, Optional, Union


@dataclass
class GanttChartData:
    mini_batch_idx: int = 0
    train_time_duration_ms: float = 0.0
    head_fwd_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    head_fwd_send_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    server_fwd_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    server_fwd_send_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    server_fwd_recv_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    tail_fwd_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    tail_bwd_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    tail_fwd_recv_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    tail_bwd_send_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    head_bwd_recv_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    server_bwd_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    server_bwd_send_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    server_bwd_recv_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    head_bwd_timestamp: List[float] = field(default_factory=lambda: [0] * 2)
    # 新增 Offload / Reload 时间字段，防止绘图时报错
    head_m_offload_ts: List[float] = field(default_factory=lambda: [0] * 2)
    tail_m_offload_ts: List[float] = field(default_factory=lambda: [0] * 2)
    head_optimizer_offload_ts: List[float] = field(default_factory=lambda: [0] * 2)
    tail_optimizer_offload_ts: List[float] = field(default_factory=lambda: [0] * 2)
    activation_offload_ts: List[float] = field(default_factory=lambda: [0] * 2)

    head_m_reload_ts: List[float] = field(default_factory=lambda: [0] * 2)
    tail_m_reload_ts: List[float] = field(default_factory=lambda: [0] * 2)
    head_optimizer_reload_ts: List[float] = field(default_factory=lambda: [0] * 2)
    tail_optimizer_reload_ts: List[float] = field(default_factory=lambda: [0] * 2)
    activation_reload_ts: List[float] = field(default_factory=lambda: [0] * 2)


def merge_cs_json(server_data: List[Dict], client_data: List[Dict], save_fp: str = "merged.json", save: bool = False) -> List[Dict]:
    # 使用字典形式合并每个 mini_batch_idx 对应的数据

    for server_item, client_item in zip(server_data, client_data):
        # 通过 mini_batch_idx 进行合并
        if server_item["mini_batch_idx"] == client_item["mini_batch_idx"]:
            # 合并：server 的非空数据覆盖 client 的空数据
            client_item["server_fwd_timestamp"] = server_item["server_fwd_timestamp"]
            client_item["server_bwd_timestamp"] = server_item["server_bwd_timestamp"]

        # 将合并后的数据写入 JSON 文件
    if save:
        with open(save_fp, "w") as f:
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
                    if v is not None and v != 0.0:
                        all_vals.append(v)

    if not all_vals:
        return []

    min_val = min(all_vals)
    # print(f"全局最小值：{min_val}")

    aligned_list: List[Dict[str, List[Optional[int]]]] = []
    for data in data_list:
        aligned: Dict[str, List[Optional[int]]] = {}
        for field_name, value in data.items():
            if isinstance(value, list):
                new_list = []
                for v in value:
                    if v is None or v == 0.0:
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
    with open(fp, "w") as f:
        json.dump(gantt_data_dict, f, indent=4)


HEAD_OFFLOAD_COLOR = "#66c2a5"
TAIL_OFFLOAD_COLOR = "#435fc2"
HEAD_RELOAD_COLOR = "#fc8d62"
TAIL_RELOAD_COLOR = "#cddc66"

# 阶段名字和颜色
STAGE_COLOR = {
    "head_fwd_timestamp": ("(C)Head Fwd", "#1f77b4"),  # 蓝色
    "head_fwd_send_timestamp": ("(C)Head Fwd Send", "#ff7f0e"),  # 橙色
    "server_fwd_timestamp": ("(S)Server Fwd", "#2ca02c"),  # 绿色
    "server_fwd_send_timestamp": ("(S)Server Fwd Send", "#dfeb56"),  # 红色
    "tail_fwd_timestamp": ("(C)Tail Fwd", "#9467bd"),  # 紫色
    "tail_bwd_timestamp": ("(C)Tail Bwd", "#8c564b"),  # 棕色
    "tail_bwd_send_timestamp": ("(C)Tail Bwd Send", "#e377c2"),  # 粉色
    "server_bwd_timestamp": ("(S)Server Bwd", "#7f7f7f"),  # 灰色
    "server_bwd_send_timestamp": ("(S)Server Bwd Send", "#bcbd22"),  # 黄绿色
    "head_bwd_timestamp": ("(C)Head Bwd", "#17becf"),  # 青色
    # "head_m_offload_ts": ("(C)Head M Offload", HEAD_OFFLOAD_COLOR),
    # "tail_m_offload_ts": ("(C)Tail M Offload", TAIL_OFFLOAD_COLOR),
    # "head_optimizer_offload_ts": ("(C)Head Opt Offload", HEAD_OFFLOAD_COLOR),
    # "tail_optimizer_offload_ts": ("(C)Tail Opt Offload", TAIL_OFFLOAD_COLOR),
    # "head_m_reload_ts": ("(C)Head M Reload", HEAD_RELOAD_COLOR),
    # "tail_m_reload_ts": ("(C)Tail M Reload", TAIL_RELOAD_COLOR),
    "head_optimizer_reload_ts": ("(C)Head Opt Reload", HEAD_RELOAD_COLOR),
    "tail_optimizer_reload_ts": ("(C)Tail Opt Reload", TAIL_RELOAD_COLOR),
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


def plot_grouped_gantt(
    mini_batch_time_gantt: List[List[GanttChartData]],
    fp: str,
    alpha: float = 0.5,
    show: bool = False,
    align: bool = True,
):
    """
    数据结构：
        mini_batch_time_gantt[0]  -> Client 的 List[GanttChartData]
        mini_batch_time_gantt[1:] -> 各个 Server rank 的 List[GanttChartData]

    行顺序：
        Client Compute
        Client Send
        Client Recv (debug only,optional)
        Server-rank0 Compute
        Server-rank1 Compute
        ...
        Server-rankN Compute
        Server Send
    """
    if not mini_batch_time_gantt:
        print("没有数据")
        return

    # ---------- 1. 展平成“扁平的 List[Dict]”，并加上 rank 信息 ----------
    flat_list: List[Dict[str, Any]] = []

    for rank_idx, rank_list in enumerate(mini_batch_time_gantt):
        if not rank_list:
            continue
        for mb in rank_list:
            if mb is None:
                continue
            d = asdict(mb)
            d["rank"] = rank_idx  # 0 = client, 1..N = server ranks
            flat_list.append(d)

    if not flat_list:
        print("没有有效的数据（全是 None？）")
        return

    # ---------- 2. 对齐时间（可选） ----------
    aligned_list = _to_aligned_ms(flat_list) if align else flat_list

    # ---------- 3. 只用我们要画的 key 来统计 all_times ----------
    CLIENT_COMPUTE_KEYS = [
        "head_fwd_timestamp",
        "head_bwd_timestamp",
        "tail_fwd_timestamp",
        "tail_bwd_timestamp",
    ]
    CLIENT_SEND_KEYS = [
        "head_fwd_send_timestamp",
        "tail_bwd_send_timestamp",
    ]
    # CLIENT_RECV_KEYS = [
    #     "head_bwd_recv_timestamp",
    #     "tail_fwd_recv_timestamp",
    # ]
    SERVER_COMPUTE_KEYS = [
        "server_fwd_timestamp",
        "server_bwd_timestamp",
    ]
    SERVER_SEND_KEYS = [
        "server_fwd_send_timestamp",
        "server_bwd_send_timestamp",
    ]

    # PLOT_KEYS = CLIENT_COMPUTE_KEYS + CLIENT_SEND_KEYS+CLIENT_RECV_KEYS + SERVER_COMPUTE_KEYS + SERVER_SEND_KEYS
    PLOT_KEYS = CLIENT_COMPUTE_KEYS + CLIENT_SEND_KEYS + SERVER_COMPUTE_KEYS + SERVER_SEND_KEYS
    all_times = []
    for aligned in aligned_list:
        for k in PLOT_KEYS:
            v = aligned.get(k)
            if not v or len(v) < 2:
                continue
            if v[0] is None or v[1] is None:
                continue
            all_times.extend(v)

    if not all_times:
        print("没有有效的时间戳")
        return

    # 统一时间零点（基于已经转成 ms 的整数）
    min_time = min(all_times)
    for aligned in aligned_list:
        for k, v in aligned.items():
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                continue
            aligned[k] = [t - min_time if isinstance(t, (int, float)) else None for t in v]

    fig, ax = plt.subplots(figsize=(15, 4))

    # ---------- 4. 构造行顺序 ----------
    # server ranks: 所有 rank > 0 的集合
    server_ranks = sorted({aligned.get("rank", 0) for aligned in aligned_list if aligned.get("rank", 0) > 0})

    # rows: (行名, 行类型, server_rank_or_None)
    # 行类型: client_compute / client_send / server_compute / server_send
    rows = []
    rows.append(("Client Compute", "client_compute", None))
    rows.append(("Client Send", "client_send", None))
    # rows.append(("Client Recv", "client_recv", None))
    for r in server_ranks:
        rows.append((f"Server-rank{r-1} Compute", "server_compute", r))
    rows.append(("Server Send", "server_send", None))
    rows.reverse()

    y_labels = [name for name, _, _ in rows]

    # ---------- 5. 画图 ----------
    used_labels = set()

    for aligned in aligned_list:
        mb_idx = aligned.get("mini_batch_idx", 0)
        rank_idx = aligned.get("rank", 0)

        for row_idx, (row_name, row_type, row_rank) in enumerate(rows):
            # Client 行：只画 rank == 0 的数据
            if row_type.startswith("client") and rank_idx != 0:
                continue
            # Server 行：只画 rank > 0 的数据
            if row_type.startswith("server") and rank_idx == 0:
                continue
            # Server Compute 行：只画对应 rank 的
            if row_type == "server_compute" and row_rank is not None and rank_idx != row_rank:
                continue

            if row_type == "client_compute":
                keys = CLIENT_COMPUTE_KEYS
            elif row_type == "client_send":
                keys = CLIENT_SEND_KEYS
            # elif row_type == "client_recv":
            #     keys = CLIENT_RECV_KEYS
            elif row_type == "server_compute":
                keys = SERVER_COMPUTE_KEYS
            elif row_type == "server_send":
                keys = SERVER_SEND_KEYS
            else:
                continue

            for key in keys:
                interval = aligned.get(key)
                if not interval or len(interval) < 2 or interval[0] is None or interval[1] is None or interval[0] == interval[1]:
                    continue

                start, end = interval
                duration = end - start
                stage_name, color = STAGE_COLOR.get(key, ("Unknown", "#cccccc"))

                # 控制图例：同一个 stage_name 只出现一次
                if stage_name not in used_labels:
                    plot_label = stage_name
                    used_labels.add(stage_name)
                else:
                    plot_label = ""

                ax.barh(
                    y=row_idx,
                    width=duration,
                    left=start,
                    height=0.8,  # ← 这里
                    color=color,
                    edgecolor="black",
                    alpha=alpha,
                    label=plot_label,
                )

                # 在块的中心标注 mini_batch_idx（如果想看到 rank，可以用 f"{rank_idx}:{mb_idx}"）
                x_center = start + duration / 2
                y_center = row_idx
                ax.text(
                    x_center,
                    y_center,
                    str(mb_idx),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )

    ax.set_xlabel("Time (ms, aligned)")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.margins(y=0)
    ax.set_title(f"Gantt Chart (Client & Server Ranks)(Config:{fp.split('/')[-1].split('.')[0]})")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.legend(
        fontsize=6,
        markerscale=0.6,
        loc="lower right",
        bbox_to_anchor=(1, 0.2),
        frameon=True,
        borderaxespad=0.3,
        handlelength=1.0,
    )
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(fp, dpi=200)
