import os
import random
from usl.utils.usl_gantt_plot import GanttChartData, plot_grouped_gantt


def _simulate_train_time(
    model_name='qwen/qwen3-1.7b',
    split_point=2,
    micro_batch_size: int = 1,
    micro_batch_num: int = 8,
    max_batch_size: int = 64,
    server_world_size: int = 4,
    random_jitter_bound: int = 0,
    head_fwd_time_per_mb: float = 7,
    head_bwd_time_per_mb: float = 21.32,
    server_fwd_time_per_mb: float = 18.7,
    server_bwd_time_per_mb: float = 42.4,
    tail_fwd_time_per_mb: float = 22,
    tail_bwd_time_per_mb: float = 53,
    client_offload_mb_num: int = 0,
    server_offload_mb_num: int = 0,
    head_offload_time: float = 0,
    head_reload_time: float = 0,
    tail_offload_time: float = 0,
    tail_reload_time: float = 0,
    head_acti_off_time_per_mb: float = 24,
    head_acti_reload_time_per_mb: float = 24,
    client_offload_model_state_sp_num: int = 0,
    server_acti_reload_time_per_mb: float = 24,
    server_acti_off_time_per_mb: float = 24,
    head_fwd_send_time: float = 10,
    tail_bwd_send_time: float = 10,
    server_fwd_send_time: float = 10,
    server_bwd_send_time: float = 10,
):
    # use list to do scheduling, each element is a list of two elements, [start_time, end_time]
    head_fwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    head_offload_timestamp = [0, 0]
    tail_reload_timestamp = [0, 0]
    head_bwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_ranks_fwd_timestamps = [[[0, 0] for _ in range(micro_batch_num)] for _ in range(server_world_size)]
    server_ranks_bwd_timestamps = [[[0, 0] for _ in range(micro_batch_num)] for _ in range(server_world_size)]
    tail_offload_timestamp = [0, 0]
    head_reload_timestamp = [0, 0]
    tail_fwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    tail_bwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    head_activation_send_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    tail_gradient_send_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_activation_send_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_gradient_send_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    # do simulating
    # step1 : do head fwd and activation offload
    for i in range(micro_batch_num):
        if i == 0:
            head_fwd_timestamps[0][1] = head_fwd_timestamps[0][0] + head_fwd_time_per_mb * (
                1 + (1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01)
            )
        else:
            head_fwd_timestamps[i][0] = head_fwd_timestamps[i - 1][1]
        head_fwd_timestamps[i][1] = head_fwd_timestamps[i][0] + max(
            head_fwd_time_per_mb, head_acti_off_time_per_mb if i < client_offload_mb_num else 0
        )
    # step 1.1 do model state offload if needed
    # if offload:
    head_offload_timestamp[0] = head_fwd_timestamps[-1][1]
    head_offload_timestamp[1] = head_offload_timestamp[0] + head_offload_time
    tail_reload_timestamp[0] = head_fwd_timestamps[-1][1]
    tail_reload_timestamp[1] = tail_reload_timestamp[0] + tail_reload_time
    # step2 : do head activation send
    for i in range(micro_batch_num):
        if i == 0:
            head_activation_send_timestamps[0][0] = head_fwd_timestamps[0][1]
        else:
            head_activation_send_timestamps[i][0] = max(head_fwd_timestamps[i][1], head_activation_send_timestamps[i - 1][1])
        head_activation_send_timestamps[i][1] = head_activation_send_timestamps[i][0] + head_fwd_send_time * (
            1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
        )
    # step3 : do server ranks fwd
    for i in range(micro_batch_num):
        for rk in range(server_world_size):
            if rk == 0:
                if i == 0:
                    server_ranks_fwd_timestamps[rk][0][0] = head_activation_send_timestamps[0][1]
                else:
                    if i > 1:
                        pre_mb_idx = i - 2
                        if pre_mb_idx < client_offload_mb_num:
                            server_ranks_fwd_timestamps[rk][i][0] = max(
                                head_activation_send_timestamps[i][1],
                                server_ranks_fwd_timestamps[rk][i - 1][1],
                                server_ranks_fwd_timestamps[rk][pre_mb_idx][1] + server_acti_off_time_per_mb,
                            )
                        else:
                            server_ranks_fwd_timestamps[rk][i][0] = max(
                                head_activation_send_timestamps[i][1], server_ranks_fwd_timestamps[rk][i - 1][1]
                            )
                    else:
                        server_ranks_fwd_timestamps[rk][i][0] = max(head_activation_send_timestamps[i][1], server_ranks_fwd_timestamps[rk][i - 1][1])
                server_ranks_fwd_timestamps[rk][i][1] = server_ranks_fwd_timestamps[rk][i][0] + server_fwd_time_per_mb * (
                    1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
                )
            else:
                if i == 0:
                    server_ranks_fwd_timestamps[rk][0][0] = server_ranks_fwd_timestamps[rk - 1][0][1]
                else:
                    if i > 1:
                        pre_mb_idx = i - 2
                        if pre_mb_idx < client_offload_mb_num:
                            server_ranks_fwd_timestamps[rk][i][0] = max(
                                server_ranks_fwd_timestamps[rk][i - 1][1],
                                server_ranks_fwd_timestamps[rk][pre_mb_idx][1] + server_acti_off_time_per_mb,
                                server_ranks_fwd_timestamps[rk - 1][i][1],
                            )
                        else:
                            server_ranks_fwd_timestamps[rk][i][0] = max(
                                server_ranks_fwd_timestamps[rk][i - 1][1], server_ranks_fwd_timestamps[rk - 1][i][1]
                            )
                    else:
                        server_ranks_fwd_timestamps[rk][i][0] = max(
                            server_ranks_fwd_timestamps[rk - 1][i][1], server_ranks_fwd_timestamps[rk][i - 1][1]
                        )
                server_ranks_fwd_timestamps[rk][i][1] = server_ranks_fwd_timestamps[rk][i][0] + server_fwd_time_per_mb * (
                    1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
                )

    # step4 : do server last rank activation send
    for i in range(micro_batch_num):
        if i == 0:
            server_activation_send_timestamps[0][0] = server_ranks_fwd_timestamps[-1][0][1]
        else:
            server_activation_send_timestamps[i][0] = max(server_ranks_fwd_timestamps[-1][i][1], server_activation_send_timestamps[i - 1][1])
        server_activation_send_timestamps[i][1] = server_activation_send_timestamps[i][0] + server_fwd_send_time * (
            1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
        )
    # step5 : do tail fwd and bwd
    for i in range(micro_batch_num):
        if i == 0:
            tail_fwd_timestamps[0][0] = max(
                head_fwd_timestamps[-1][1] + head_offload_time,
                head_fwd_timestamps[-1][1] + tail_reload_time,
                server_activation_send_timestamps[0][1],
            ) + (head_acti_reload_time_per_mb if client_offload_mb_num > 0 else 0)
        else:
            tail_fwd_timestamps[i][0] = max(tail_bwd_timestamps[i - 1][1], server_activation_send_timestamps[i][1])
        tail_fwd_timestamps[i][1] = tail_fwd_timestamps[i][0] + tail_fwd_time_per_mb * (
            1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
        )
        tail_bwd_timestamps[i][0] = tail_fwd_timestamps[i][1]
        tail_bwd_timestamps[i][1] = tail_bwd_timestamps[i][0] + tail_bwd_time_per_mb * (
            1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
        )

    # step6 : do client grad send to server
    for i in range(micro_batch_num):
        if i == 0:
            tail_gradient_send_timestamps[0][0] = max(head_activation_send_timestamps[-1][1], tail_bwd_timestamps[0][1])
        else:
            tail_gradient_send_timestamps[i][0] = max(tail_gradient_send_timestamps[i - 1][1], tail_bwd_timestamps[i][1])
        tail_gradient_send_timestamps[i][1] = tail_gradient_send_timestamps[i][0] + tail_bwd_send_time * (
            1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
        )

    # step7 : do server bwd
    for i in range(micro_batch_num):
        for rk in range(server_world_size - 1, -1, -1):
            if rk == server_world_size - 1:
                if i == 0:
                    server_ranks_bwd_timestamps[rk][0][0] = max(
                        server_ranks_fwd_timestamps[rk][-1][1], tail_gradient_send_timestamps[0][1], server_activation_send_timestamps[-1][1]
                    ) + (server_acti_reload_time_per_mb if server_offload_mb_num > 0 else 0)
                else:
                    if i < server_offload_mb_num:
                        server_ranks_bwd_timestamps[rk][i][0] = max(
                            server_ranks_bwd_timestamps[rk][i - 1][1],
                            server_ranks_bwd_timestamps[rk][i - 1][0] + server_acti_reload_time_per_mb,
                            tail_gradient_send_timestamps[i][1],
                        )
                    else:
                        server_ranks_bwd_timestamps[rk][i][0] = max(
                            server_ranks_bwd_timestamps[rk][i - 1][1],
                            tail_gradient_send_timestamps[i][1],
                        )

                server_ranks_bwd_timestamps[rk][i][1] = server_ranks_bwd_timestamps[rk][i][0] + server_bwd_time_per_mb * (
                    1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
                )
            else:
                if i == 0:
                    server_ranks_bwd_timestamps[rk][0][0] = server_ranks_bwd_timestamps[rk + 1][0][1] + (
                        server_acti_reload_time_per_mb if server_offload_mb_num > 0 else 0
                    )
                else:
                    if i < server_offload_mb_num:
                        server_ranks_bwd_timestamps[rk][i][0] = max(
                            server_ranks_bwd_timestamps[rk][i - 1][1],
                            server_ranks_bwd_timestamps[rk][i - 1][0] + server_acti_reload_time_per_mb,
                            server_ranks_bwd_timestamps[rk + 1][i][1],
                        )
                    else:
                        server_ranks_bwd_timestamps[rk][i][0] = max(
                            server_ranks_bwd_timestamps[rk][i - 1][1], server_ranks_bwd_timestamps[rk + 1][i][1]
                        )
                server_ranks_bwd_timestamps[rk][i][1] = server_ranks_bwd_timestamps[rk][i][0] + server_bwd_time_per_mb * (
                    1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
                )

    # step8 : do server grad send to head
    for i in range(micro_batch_num):
        if i == 0:
            server_gradient_send_timestamps[0][0] = max(server_activation_send_timestamps[-1][1], server_ranks_bwd_timestamps[0][0][1])
        else:
            server_gradient_send_timestamps[i][0] = max(server_gradient_send_timestamps[i - 1][1], server_ranks_bwd_timestamps[0][i][1])
        server_gradient_send_timestamps[i][1] = server_gradient_send_timestamps[i][0] + server_bwd_send_time * (
            1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
        )

    # if offload:
    head_reload_timestamp[0] = head_fwd_timestamps[-1][1]
    head_reload_timestamp[1] = head_offload_timestamp[0] + head_reload_time
    tail_offload_timestamp[0] = head_fwd_timestamps[-1][1]
    tail_offload_timestamp[1] = tail_offload_timestamp[0] + tail_offload_time
    # step9 : do head bwd
    for i in range(micro_batch_num):
        if i == 0:
            head_bwd_timestamps[0][0] = max(
                tail_bwd_timestamps[-1][1] + tail_offload_time,
                tail_bwd_timestamps[-1][1] + head_reload_time,
                server_gradient_send_timestamps[0][1],
            ) + (head_acti_reload_time_per_mb if i < client_offload_mb_num else 0)
            head_bwd_timestamps[i][1] = head_bwd_timestamps[i][0] + head_bwd_time_per_mb * (
                1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
            )
        else:
            head_bwd_timestamps[i][0] = max(
                head_bwd_timestamps[i - 1][0] + head_acti_reload_time_per_mb,
                head_bwd_timestamps[i - 1][1],
                server_gradient_send_timestamps[i][1],
            )
            head_bwd_timestamps[i][1] = head_bwd_timestamps[i][0] + max(
                head_bwd_time_per_mb, head_acti_reload_time_per_mb if i < client_offload_mb_num else 0
            )
    batch_train_time = head_bwd_timestamps[-1][1] - head_fwd_timestamps[0][0]
    # print(head_fwd_timestamps)
    gantt_data = [[GanttChartData(mini_batch_idx=i) for i in range(micro_batch_num)] for _ in range(server_world_size + 1)]
    # gantt_data[0] is client data
    # gantt_data[1:server_world_size+1] is server data
    client_data = gantt_data[0]
    for i in range(micro_batch_num):
        client_data[i].train_time_duration_ms = head_bwd_timestamps[i][1] - head_fwd_timestamps[i][0]
        client_data[i].head_fwd_timestamp = head_fwd_timestamps[i]
        client_data[i].head_fwd_send_timestamp = head_activation_send_timestamps[i]
        client_data[i].tail_fwd_timestamp = tail_fwd_timestamps[i]
        client_data[i].tail_bwd_timestamp = tail_bwd_timestamps[i]
        client_data[i].tail_bwd_send_timestamp = tail_gradient_send_timestamps[i]
        client_data[i].head_bwd_timestamp = head_bwd_timestamps[i]
    server_data = gantt_data[1:]
    for rk in range(server_world_size):
        for i in range(micro_batch_num):
            if rk == 0:
                server_data[rk][i].server_bwd_send_timestamp = server_gradient_send_timestamps[i]
            if rk == server_world_size - 1:
                server_data[rk][i].server_fwd_send_timestamp = server_activation_send_timestamps[i]
            server_data[rk][i].server_fwd_timestamp = server_ranks_fwd_timestamps[rk][i]
            server_data[rk][i].server_bwd_timestamp = server_ranks_bwd_timestamps[rk][i]
    save_dir = f'log/img/simulated/{model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fp = (
        f'{save_dir}/sp_{split_point}_b_{micro_batch_num}_mb_{micro_batch_size}_s_{512}_mbps_{230}_pipedream_wc'
        f'{f"_coa_{client_offload_mb_num}_cos_{client_offload_model_state_sp_num}_soa_{server_offload_mb_num}"}.png'
    )
    plot_grouped_gantt(gantt_data, fp, align=False)
    batch_train_time = head_bwd_timestamps[-1][1] - head_fwd_timestamps[0][0]
    print(f'batch_train_time: {batch_train_time} ms')


_simulate_train_time()
