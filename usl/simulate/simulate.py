import argparse

import math
import random
import os
import json
import time
from typing import Dict, List
from usl.utils.usl_gantt_plot import GanttChartData, plot_grouped_gantt
from usl.simulate import *
from usl.simulate.profile_separate_nodes import run_profile
import pandas as pd
from dataclasses import dataclass


def _simulate_train_time(
    main_var: MainVariable,
    time_const: TimeConstant,
    mem_const: MemoryConstant,
    curr_obj: Objective,
    save_gantt: bool = False,
    save_sim_res: bool = False,
) -> SimulateResult:
    # simulate the batch training time
    assert (
        curr_obj.min_gpu_num_required > 0
    ), "min_gpu_num_required should be greater than 0,must do memory simulation first, see method _simulate_memory_usage"
    gpu_num_required = curr_obj.min_gpu_num_required
    layers_per_gpu_list = curr_obj.layers_num_per_gpu
    total_layer_num = sum(layers_per_gpu_list)
    # ------------------------get meta time variables to simulate the batch training time ----------
    random_jitter_bound = 1  # random jitter bound for the batch size
    micro_batch_num = (main_var.batch_size + mem_const.micro_batch_size - 1) // mem_const.micro_batch_size
    split_point = main_var.split_point
    max_split_point = mem_const.max_split_point
    base_split_point = mem_const.baseline_split_point
    ms_offload_sp_num = main_var.client_offload_model_state_sp_num
    assert split_point > 0, "split_point should be greater than 0"
    head_fwd_time_per_mb = time_const.base_head_fwd_time_per_mb + (split_point - base_split_point) * time_const.head_fwd_time_increment_per_sp
    head_bwd_time_per_mb = time_const.base_head_bwd_time_per_mb + (split_point - base_split_point) * time_const.head_bwd_time_increment_per_sp
    server_fwd_time_per_mb_per_rank = [
        (time_const.base_server_fwd_time_per_mb + (max_split_point - split_point - base_split_point) * time_const.server_fwd_time_increment_per_sp)
        / total_layer_num
        * layer_num
        for layer_num in layers_per_gpu_list
    ]
    # print(server_fwd_time_per_mb_per_rank)
    server_bwd_time_per_mb_per_rank = [
        (time_const.base_server_bwd_time_per_mb + (max_split_point - split_point - base_split_point) * time_const.server_bwd_time_increment_per_sp)
        / total_layer_num
        * layer_num
        for layer_num in layers_per_gpu_list
    ]
    tail_fwd_time_per_mb = time_const.base_tail_fwd_time_per_mb + (split_point - base_split_point) * time_const.tail_fwd_time_increment_per_sp
    tail_bwd_time_per_mb = time_const.base_tail_bwd_time_per_mb + (split_point - base_split_point) * time_const.tail_bwd_time_increment_per_sp
    head_offload_time = (
        time_const.base_head_model_state_offload_time + (ms_offload_sp_num - base_split_point) * time_const.head_model_offload_time_increment_per_sp
    )
    head_reload_time = head_offload_time
    tail_offload_time = (
        time_const.base_tail_model_state_offload_time + (ms_offload_sp_num - base_split_point) * time_const.tail_model_offload_time_increment_per_sp
    )
    tail_reload_time = tail_offload_time
    head_acti_off_time_per_mb = (
        time_const.base_head_activation_offload_time_per_mb
        + (split_point - base_split_point) * time_const.head_activation_offload_time_increment_per_sp
    )
    head_acti_reload_time_per_mb = (
        time_const.base_head_activation_reload_time_per_mb
        + (split_point - base_split_point) * time_const.head_activation_reload_time_increment_per_sp
    )
    server_acti_off_time_per_mb_per_rank = [
        (
            time_const.base_server_activation_offload_time_per_mb
            + (split_point - base_split_point) * time_const.server_activation_offload_time_increment_per_sp
        )
        / total_layer_num
        * layer_num
        for layer_num in layers_per_gpu_list
    ]
    server_acti_reload_time_per_mb_per_rank = [
        (
            time_const.base_server_activation_reload_time_per_mb
            + (split_point - base_split_point) * time_const.server_activation_reload_time_increment_per_sp
        )
        / total_layer_num
        * layer_num
        for layer_num in layers_per_gpu_list
    ]
    network_delay_time = 10
    head_fwd_send_time = time_const.head_activation_send_time + network_delay_time
    server_fwd_send_time = time_const.server_activation_send_time + network_delay_time
    tail_bwd_send_time = time_const.tail_gradient_send_time + network_delay_time
    server_bwd_send_time = time_const.server_gradient_send_time + network_delay_time

    # use list to do scheduling, each element is a list of two elements, [start_time, end_time]
    head_fwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    head_activation_offload_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    head_activation_reload_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    head_offload_timestamp = [0, 0]
    tail_reload_timestamp = [0, 0]
    head_bwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    # 将服务端改成多卡
    # server_fwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    # server_bwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_activation_offload_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_activation_reload_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_ranks_fwd_timestamps = [[[0, 0] for _ in range(micro_batch_num)] for _ in range(gpu_num_required)]
    server_ranks_bwd_timestamps = [[[0, 0] for _ in range(micro_batch_num)] for _ in range(gpu_num_required)]
    server_ranks_activation_offload_timestamps = [[[0, 0] for _ in range(micro_batch_num)] for _ in range(gpu_num_required)]
    server_ranks_activation_reload_timestamps = [[[0, 0] for _ in range(micro_batch_num)] for _ in range(gpu_num_required)]
    tail_offload_timestamp = [0, 0]
    head_reload_timestamp = [0, 0]
    tail_fwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    tail_bwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    head_activation_send_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    tail_gradient_send_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_activation_send_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_gradient_send_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    # print(acti_reload_time_per_mb)
    # time_const.delay_time_avg_ms = 40
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
            head_fwd_time_per_mb, head_acti_off_time_per_mb if i < main_var.client_offload_mb_num else 0
        )
    # step 1.1 do model state offload if needed
    # if main_var.offload:
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
        for rk in range(gpu_num_required):
            if rk == 0:
                if i == 0:
                    server_ranks_fwd_timestamps[rk][0][0] = head_activation_send_timestamps[0][1]
                else:
                    if i > 1:
                        pre_mb_idx = i - 2
                        if pre_mb_idx < main_var.client_offload_mb_num:
                            server_ranks_fwd_timestamps[rk][i][0] = max(
                                head_activation_send_timestamps[i][1],
                                server_ranks_fwd_timestamps[rk][i - 1][1],
                                server_ranks_fwd_timestamps[rk][pre_mb_idx][1] + server_acti_off_time_per_mb_per_rank[rk],
                            )
                        else:
                            server_ranks_fwd_timestamps[rk][i][0] = max(
                                head_activation_send_timestamps[i][1], server_ranks_fwd_timestamps[rk][i - 1][1]
                            )
                    else:
                        server_ranks_fwd_timestamps[rk][i][0] = max(head_activation_send_timestamps[i][1], server_ranks_fwd_timestamps[rk][i - 1][1])
                server_ranks_fwd_timestamps[rk][i][1] = server_ranks_fwd_timestamps[rk][i][0] + server_fwd_time_per_mb_per_rank[rk] * (
                    1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
                )
            else:
                if i == 0:
                    server_ranks_fwd_timestamps[rk][0][0] = server_ranks_fwd_timestamps[rk - 1][0][1]
                else:
                    if i > 1:
                        pre_mb_idx = i - 2
                        if pre_mb_idx < main_var.client_offload_mb_num:
                            server_ranks_fwd_timestamps[rk][i][0] = max(
                                server_ranks_fwd_timestamps[rk][i - 1][1],
                                server_ranks_fwd_timestamps[rk][pre_mb_idx][1] + server_acti_off_time_per_mb_per_rank[rk],
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
                server_ranks_fwd_timestamps[rk][i][1] = server_ranks_fwd_timestamps[rk][i][0] + server_fwd_time_per_mb_per_rank[rk] * (
                    1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
                )

            # if i == micro_batch_num - 1 and micro_batch_num == main_var.client_offload_mb_num:
            #     server_ranks_fwd_timestamps[rk][i][1] += server_acti_off_time_per_mb

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
            ) + (head_acti_reload_time_per_mb if main_var.client_offload_mb_num > 0 else 0)
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
        for rk in range(gpu_num_required - 1, -1, -1):
            if rk == gpu_num_required - 1:
                if i == 0:
                    server_ranks_bwd_timestamps[rk][0][0] = max(server_ranks_fwd_timestamps[rk][-1][1], tail_gradient_send_timestamps[0][1]) + (
                        server_acti_reload_time_per_mb_per_rank[rk] if main_var.server_offload_mb_num > 0 else 0
                    )
                else:
                    if i < main_var.server_offload_mb_num:
                        server_ranks_bwd_timestamps[rk][i][0] = max(
                            server_ranks_bwd_timestamps[rk][i - 1][1],
                            server_ranks_bwd_timestamps[rk][i - 1][0] + server_acti_reload_time_per_mb_per_rank[rk],
                            tail_gradient_send_timestamps[i][1],
                        )
                    else:
                        server_ranks_bwd_timestamps[rk][i][0] = max(
                            server_ranks_bwd_timestamps[rk][i - 1][1],
                            tail_gradient_send_timestamps[i][1],
                        )

                server_ranks_bwd_timestamps[rk][i][1] = server_ranks_bwd_timestamps[rk][i][0] + server_bwd_time_per_mb_per_rank[rk] * (
                    1 + random.randint(-random_jitter_bound, random_jitter_bound) * 0.01
                )
            else:
                if i == 0:
                    server_ranks_bwd_timestamps[rk][0][0] = server_ranks_bwd_timestamps[rk + 1][0][1] + (
                        server_acti_reload_time_per_mb_per_rank[rk] if main_var.server_offload_mb_num > 0 else 0
                    )
                else:
                    if i < main_var.server_offload_mb_num:
                        server_ranks_bwd_timestamps[rk][i][0] = max(
                            server_ranks_bwd_timestamps[rk][i - 1][1],
                            server_ranks_bwd_timestamps[rk][i - 1][0] + server_acti_reload_time_per_mb_per_rank[rk],
                            server_ranks_bwd_timestamps[rk + 1][i][1],
                        )
                    else:
                        server_ranks_bwd_timestamps[rk][i][0] = max(
                            server_ranks_bwd_timestamps[rk][i - 1][1], server_ranks_bwd_timestamps[rk + 1][i][1]
                        )
                server_ranks_bwd_timestamps[rk][i][1] = server_ranks_bwd_timestamps[rk][i][0] + server_bwd_time_per_mb_per_rank[rk] * (
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

    # if main_var.offload:
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
            ) + (head_acti_reload_time_per_mb if i < main_var.client_offload_mb_num else 0)
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
                head_bwd_time_per_mb, head_acti_reload_time_per_mb if i < main_var.client_offload_mb_num else 0
            )
    # print(head_fwd_timestamps)
    if save_gantt or save_sim_res:
        gantt_data = [[GanttChartData(mini_batch_idx=i) for i in range(micro_batch_num)] for _ in range(gpu_num_required + 1)]
        # gantt_data[0] is client data
        # gantt_data[1:gpu_num_required+1] is server data
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
        for rk in range(gpu_num_required):
            for i in range(micro_batch_num):
                if rk == 0:
                    server_data[rk][i].server_bwd_send_timestamp = server_gradient_send_timestamps[i]
                if rk == gpu_num_required - 1:
                    server_data[rk][i].server_fwd_send_timestamp = server_activation_send_timestamps[i]
                server_data[rk][i].server_fwd_timestamp = server_ranks_fwd_timestamps[rk][i]
                server_data[rk][i].server_bwd_timestamp = server_ranks_bwd_timestamps[rk][i]
        if save_gantt:
            save_dir = f'log/img/simulated/{model_name}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fp = (
                f'{save_dir}/sp_{split_point}_b_{main_var.batch_size}_mb_{mem_const.micro_batch_size}_s_{mem_const.max_seq_len}_mbps_{mbps}_pipedream_wc{"_lora" if lora else ""}'
                f'{f"_coa_{main_var.client_offload_mb_num}_cos_{main_var.client_offload_model_state_sp_num}_soa_{main_var.server_offload_mb_num}"}.png'
            )
            plot_grouped_gantt(gantt_data, fp, align=False)
        if save_sim_res:
            gantt_data_dict = {
                "model_name": model_name,
                "layer_num": total_layer_num,
                "split_point": split_point,
                "micro_batch_size": mem_const.micro_batch_size,
                "micro_batch_num": micro_batch_num,
                "max_batch_size": max_batch_size,
                "server_world_size": gpu_num_required,
                "random_jitter_bound": random_jitter_bound,
                "head_fwd_time_per_mb": head_fwd_time_per_mb,
                "head_bwd_time_per_mb": head_bwd_time_per_mb,
                "server_fwd_time_per_mb_per_rank": server_fwd_time_per_mb_per_rank,
                "server_bwd_time_per_mb_per_rank": server_bwd_time_per_mb_per_rank,
                "tail_fwd_time_per_mb": tail_fwd_time_per_mb,
                "tail_bwd_time_per_mb": tail_bwd_time_per_mb,
                "head_offload_time": head_offload_time,
                "head_reload_time": head_reload_time,
                "tail_offload_time": tail_offload_time,
                "tail_reload_time": tail_reload_time,
                "head_acti_off_time_per_mb": head_acti_off_time_per_mb,
                "head_acti_reload_time_per_mb": head_acti_reload_time_per_mb,
                "server_acti_reload_time_per_mb_per_rank": server_acti_reload_time_per_mb_per_rank,
                "server_acti_off_time_per_mb_per_rank": server_acti_off_time_per_mb_per_rank,
                "head_fwd_send_time": head_fwd_send_time,
                "tail_bwd_send_time": tail_bwd_send_time,
                "server_fwd_send_time": server_fwd_send_time,
                "server_bwd_send_time": server_bwd_send_time,
            }
            save_dir = f'log/profile/time_sim_res/{model_name}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fp = (
                f'{save_dir}/sp_{split_point}_b_{main_var.batch_size}_mb_{mem_const.micro_batch_size}_s_{mem_const.max_seq_len}_mbps_{mbps}_pipedream_wc{"_lora" if lora else ""}'
                f'{f"_coa_{main_var.client_offload_mb_num}_cos_{main_var.client_offload_model_state_sp_num}_soa_{main_var.server_offload_mb_num}"}.json'
            )
            with open(fp, "w") as f:
                json.dump(gantt_data_dict, f, indent=4)
    # calculate objective function
    # calculate batch train time ,unit:ms
    batch_train_time = head_bwd_timestamps[-1][1] - head_fwd_timestamps[0][0]
    # calculate client idle rate
    client_compute_time = 0  # unit : ms
    client_idle_rate = 100
    for i in range(micro_batch_num):
        client_compute_time += (
            head_fwd_timestamps[i][1]
            - head_fwd_timestamps[i][0]
            + tail_fwd_timestamps[i][1]
            - tail_fwd_timestamps[i][0]
            + tail_bwd_timestamps[i][1]
            - tail_bwd_timestamps[i][0]
            + head_bwd_timestamps[i][1]
            - head_bwd_timestamps[i][0]
        )
        client_idle_rate = 100 - (client_compute_time / batch_train_time) * 100
    # calculate server idle rate
    server_compute_time = 0  # unit : ms
    server_idle_rate = 100
    # TODO : calculate server idle rate
    # for i in range(micro_batch_num):
    #     server_compute_time += server_fwd_timestamps[i][1] - server_fwd_timestamps[i][0] + server_bwd_timestamps[i][1] - server_bwd_timestamps[i][0]
    #     server_idle_rate = 100 - (server_compute_time / batch_train_time) * 100
    # calculate client send rate
    client_send_time = 0  # unit : ms
    for i in range(micro_batch_num):
        client_send_time += (
            head_activation_send_timestamps[i][1]
            - head_activation_send_timestamps[i][0]
            + tail_gradient_send_timestamps[i][1]
            - tail_gradient_send_timestamps[i][0]
        )
    client_send_rate = client_send_time / (batch_train_time) * 100
    # calculate server send rate
    server_send_time = 0  # unit : ms
    for i in range(micro_batch_num):
        server_send_time += (
            server_activation_send_timestamps[i][1]
            - server_activation_send_timestamps[i][0]
            + server_gradient_send_timestamps[i][1]
            - server_gradient_send_timestamps[i][0]
        )
    server_send_rate = server_send_time / (batch_train_time) * 100
    curr_obj.client_idle_rate = round(client_idle_rate, 2)
    curr_obj.server_idle_rate = round(server_idle_rate, 2)
    curr_obj.client_send_rate = round(client_send_rate, 2)
    curr_obj.server_send_rate = round(server_send_rate, 2)
    curr_obj.batch_train_time = round(batch_train_time, 4)
    curr_obj.epoch_train_time = round(batch_train_time * main_var.total_sample_count / main_var.batch_size / 1000 / 3600, 4)  # ms -> hours
    curr_obj.server_cost_per_epoch = round(
        curr_obj.min_gpu_num_required * curr_obj.epoch_train_time * main_var.gpu_rent_cost_per_hour * 100, 4
    )  # 美分
    return SimulateResult(main_variable=main_var, time_const=time_const, objective=curr_obj)


def _simulate_peak_mem_alloc(main_var: MainVariable, memory_const: MemoryConstant) -> SimulateResult:
    # simulate the peak memory allocation
    split_point = main_var.split_point
    batch_size = main_var.batch_size
    max_split_point = memory_const.max_split_point
    base_sp = memory_const.baseline_split_point  # default 1
    base_mb_num = memory_const.baseline_minibatch_num  # default 4
    client_offload_mb_num = main_var.client_offload_mb_num
    server_offload_mb_num = main_var.server_offload_mb_num
    os_offload_sp_num = main_var.client_offload_model_state_sp_num
    assert split_point >= 1 and split_point < max_split_point, "split_point should be greater than or equal to 1 and less than max_split_point"
    assert batch_size >= 4, "batch size should be greater than or equal to 4"
    assert client_offload_mb_num >= 0 and server_offload_mb_num >= 0, "client_offload_mb_num and server_offload_mb_num should be greater than 0"
    client_peak_mem_alloc = (
        memory_const.base_client_mem_alloc  # 基础的显存开销
        + (split_point - base_sp) * memory_const.mem_increment_per_sp_client  # 因为切分层数增加，显存开销增加
        + (batch_size - base_mb_num) * split_point * memory_const.mem_increment_per_sp_mb_client
        # 因为模型切分层和batch size增加，激活量引起的显存开销
        - (split_point * (max(0, client_offload_mb_num - 1)) * memory_const.mem_increment_per_sp_mb_client)  # 因激活量卸载，显存开销减少
        + client_offload_mb_num * memory_const.mem_increment_per_mb_client_if_oa
        # + (client_offload_mb_num - base_mb_num) * memory_const.mem_increment_per_mb_client_if_oa # 12/5
    )
    if os_offload_sp_num > 0:
        client_peak_mem_alloc -= (
            memory_const.base_model_state_mem_alloc_except_blocks + os_offload_sp_num * memory_const.model_mem_increment_per_sp_client
        )

    server_peak_mem_alloc = (
        memory_const.base_server_mem_alloc
        - (split_point - base_sp) * memory_const.mem_increment_per_sp_server
        + (batch_size - base_mb_num) * (max_split_point - split_point) * memory_const.mem_increment_per_sp_mb_server
        - (max_split_point - split_point)
        * (max(0, server_offload_mb_num - (2 if main_var.lora else 3)))
        * memory_const.mem_increment_per_sp_mb_server
        + server_offload_mb_num * memory_const.mem_increment_per_mb_server_if_oa
        # + (server_offload_mb_num - base_mb_num) * memory_const.mem_increment_per_mb_server_if_oa
    )

    # calculate min_gpu_num_required and mem_alloc_per_gpu
    safe_factor = 0.95  # 安全系数，防止超配
    for i in range(1, 30):
        min_gpu_num_required = i
        # add nccl buffer
        server_peak_mem_alloc_tmp = server_peak_mem_alloc + (
            memory_const.nccl_base_buffer_size + memory_const.nccl_buffer_per_rank_per_mb * main_var.batch_size
        ) * (min_gpu_num_required - 1)
        if server_peak_mem_alloc_tmp > min_gpu_num_required * main_var.gpu_max_capacity * 1024 * safe_factor:
            continue
        else:
            server_peak_mem_alloc = server_peak_mem_alloc_tmp
            server_total_layer_num = (max_split_point - split_point) * 2
            mem_required_per_layer = server_peak_mem_alloc / server_total_layer_num
            # max_layers_per_gpu = math.floor(main_var.gpu_max_capacity * safe_factor * 1024 / mem_required_per_layer)
            # 0.9 is a safety factor to avoid over-provisioning
            # min_gpu_num_required = math.ceil(server_total_layer_num / max_layers_per_gpu)
            break

    # detail layers_num_per_gpu and mem_alloc_per_gpu
    layers_num_per_gpu = [server_total_layer_num // min_gpu_num_required] * min_gpu_num_required
    i = 0
    while sum(layers_num_per_gpu) < server_total_layer_num:
        layers_num_per_gpu[i] += 1
        i = (i + 1) % min_gpu_num_required
    mem_alloc_per_gpu = [round(mem_required_per_layer * layers_num, 2) for layers_num in layers_num_per_gpu]
    return SimulateResult(
        objective=Objective(
            client_peak_mem_alloc=client_peak_mem_alloc,
            server_peak_mem_alloc=server_peak_mem_alloc,
            min_gpu_num_required=min_gpu_num_required,
            mem_alloc_per_layer=mem_required_per_layer,
            layers_num_per_gpu=layers_num_per_gpu,
            mem_alloc_per_gpu=mem_alloc_per_gpu,
        )
    )


def simulate(
    main_var: MainVariable, time_const: TimeConstant, mem_const: MemoryConstant, save_gantt: bool = False, save_time_res: bool = False
) -> SimulateResult:
    simulate_mem_result = _simulate_peak_mem_alloc(main_var, mem_const)
    simulate_result = _simulate_train_time(main_var, time_const, mem_const, simulate_mem_result.objective, save_gantt, save_time_res)
    # copy memory result to simulate_result
    # simulate_result.objective.client_peak_mem_alloc = simulate_mem_result.objective.client_peak_mem_alloc
    # simulate_result.objective.server_peak_mem_alloc = simulate_mem_result.objective.server_peak_mem_alloc
    # simulate_result.objective.server_cost_per_epoch = round(
    #     simulate_result.objective.server_peak_mem_alloc * simulate_result.objective.epoch_train_time / 10**6, 2
    # )
    return simulate_result


@dataclass
class ParetoPoint:
    cost: float
    time: float
    payload: Dict


def _non_dominated_insert(pareto_list: list[ParetoPoint], cand_cost: float, cand_time: float, cand_payload: Dict, eps_cost=0.0, eps_time=0.0):
    """
    将候选点插入帕累托集合：
    - 若被现有点(成本<=cand_cost-eps_cost 且 时间<=cand_time-eps_time)严格支配 -> 丢弃
    - 否则移除所有被候选点严格支配的点，然后加入
    """
    # 检查是否被支配
    for p in pareto_list:
        if (
            (p.cost <= cand_cost - eps_cost)
            and (p.time <= cand_time - eps_time)
            and ((p.cost < cand_cost - eps_cost) or (p.time < cand_time - eps_time))
        ):
            return  # cand 被支配，直接丢弃

    # 移除被 cand 支配的点
    new_list = []
    for p in pareto_list:
        dominated_by_cand = (
            (cand_cost <= p.cost - eps_cost)
            and (cand_time <= p.time - eps_time)
            and ((cand_cost < p.cost - eps_cost) or (cand_time < p.time - eps_time))
        )
        if not dominated_by_cand:
            new_list.append(p)

    new_list.append(ParetoPoint(cand_cost, cand_time, cand_payload))
    pareto_list[:] = new_list  # 就地更新


def do_optimize(
    model_name,
    dataset_size,
    max_split_point,
    max_batch_size,
    time_res,
    mem_res,
    max_client_mem_mb,
    lora,
):
    all_data = []
    pareto_front: List[ParetoPoint] = []  # 存 ParetoPoint
    best_strategy = None
    min_cost = float("inf")
    min_epoch_train_time = float("inf")
    # ε-容差（可按测量抖动调节，比如 0.5%）
    EPS_COST_RATIO = 0.005
    EPS_TIME_RATIO = 0.005
    # 遍历所有可能的配置
    time_start = time.time()
    # --------modify batch size----------------
    for bs in range(8, max_batch_size + 1, 2):
        # for bs in [8]:
        print(f'searching batch size {bs},time stacked: {time.time() - time_start:.4f}s')
        skip_curr_bs = False
        for sp in range(1, max_split_point + 1):
            # for sp in [4]:
            if skip_curr_bs:
                break
            # for c_omb in range(bs, 0, -1):
            #     if skip_curr_bs:
            #         break
            #     for ossp in range(sp, -1, -1):
            #         if skip_curr_bs:
            #             break
            #         for s_omb in range(bs, 2, -1):
            var = MainVariable(
                total_sample_count=dataset_size,
                batch_size=bs,
                split_point=sp,
                client_offload_mb_num=bs,
                server_offload_mb_num=bs,
                client_offload_model_state_sp_num=sp,
                lora=lora,
            )
            sim_res = simulate(var, time_res, mem_res, save_gantt=False)
            sim_res.model_name = model_name
            # mem_sim_res = _simulate_peak_mem_alloc(var, mem_res)
            if sim_res.objective.client_peak_mem_alloc > max_client_mem_mb * 0.85:
                skip_curr_bs = True
                break
            if best_strategy is None and sim_res.objective.client_peak_mem_alloc < max_client_mem_mb * 0.85:  # 0.95 is for safety
                best_strategy = sim_res
                min_cost = best_strategy.objective.server_cost_per_epoch
                min_epoch_train_time = best_strategy.objective.epoch_train_time
            # time_sim_res = _simulate_train_time(var, time_res, mem_res, save_gantt=False)
            server_cost = sim_res.objective.server_cost_per_epoch
            epoch_train_time = sim_res.objective.epoch_train_time
            # 维护单目标最优（若仍想保留）
            if server_cost < min_cost:
                best_strategy = sim_res  # 或者存一个自定义组合对象
                min_cost = server_cost
                min_epoch_train_time = epoch_train_time
            # 维护多目标最优
            eps_cost = server_cost * EPS_COST_RATIO
            eps_time = epoch_train_time * EPS_TIME_RATIO
            _non_dominated_insert(
                pareto_front,
                cand_cost=server_cost,
                cand_time=epoch_train_time,
                cand_payload=sim_res.to_simple_dict(),
                eps_cost=eps_cost,
                eps_time=eps_time,
            )
    print(f"Total time: {time.time() - time_start:.4f}s")
    # ---- 搜索结束：对帕累托前沿按成本再按时间排序，输出/返回 ----
    pareto_front.sort(key=lambda p: (p.cost, p.time))
    df = pd.DataFrame([pf.payload for pf in pareto_front])
    df = df.round(2)[:100]
    df.to_csv(
        f'log/simulate_results_{model_name.split("/")[-1]}_bs_{max_batch_size}_sp_{max_split_point}{f'_mps_{mps_gpu}' if mps_gpu<100 else ''}{'_lora' if lora else ''}.csv',
        index=False,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description='Simulate memory and time profiling with dynamic parameters.')

    # Defining the command-line arguments
    # qwen/qwen3-4b qwen/qwen3-8b qwen/qwen3-14b qwen/qwen3-32b meta-llama/llama3-8b gemma/gemma2-9b
    parser.add_argument('--model', type=str, default='qwen/qwen3-8b', help='The model name.')
    parser.add_argument('--max_client_mem_gb', type=int, default=24, help='The maximum memory allocation for the client.')
    # parser.add_argument('--max_split_point', '-MSP', type=int, default=7, help='The maximum split point for the model.')
    parser.add_argument('--max_sequence_len', '-L', type=int, default=512, help='The sample nums of dataset')
    parser.add_argument('--dataset_size', '-DS', type=int, default=12460, help='The sample nums of dataset')
    parser.add_argument('--lora', action='store_true', help='Whether to use Lora or not.')
    parser.add_argument('--mbps', type=int, default=230, help='The mbps value for the simulation.')
    parser.add_argument('--mps_gpu', type=int, default=100, help='The max percentage of GPU active threads used for the simulation.')
    parser.add_argument('--max_batch_size', '-BS', type=int, default=64, help='The max batch size for the simulation.')
    parser.add_argument('--profile_dir', type=str, default='log/profile/sim_profile', help='The profile directory for storing results.')
    parser.add_argument('--save_gantt', action='store_true', help='Whether to save gantt chart or not.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_name = args.model
    # lora = args.lora
    lora = True  # always use lora
    mbps = args.mbps
    max_batch_size = args.max_batch_size
    profile_dir = args.profile_dir
    # max_split_point = args.max_split_point
    dataset_size = args.dataset_size
    max_client_mem_mb = args.max_client_mem_gb * 1024
    save_gantt = args.save_gantt
    max_sequence_len = args.max_sequence_len
    mps_gpu = args.mps_gpu
    # run profiling
    profile_start = time.time()
    mem_res, time_res = run_profile(model_name, mbps, lora, base_sp=2, base_bs=8, mps_gpu=mps_gpu, profile_dir=profile_dir)
    print(f"Profiling time: {time.time() - profile_start:.4f}s")
    if mem_res is None or time_res is None:
        print("Failed to run profiling.")
        exit(1)
    # print(mem_res)
    for key, value in mem_res.__dict__.items():
        print(key, value)
    for key, value in time_res.__dict__.items():
        print(key, value)
    all_data = []
    # for sp in [1, 2, 3, 4, 5, 6]:  # 按照模型层数的一半去设
    for sp in range(1, 16):
        # for bs in [8, 16, 32]:
        for bs in range(8, max_batch_size + 1, 2):
            var = MainVariable(
                total_sample_count=12460,
                batch_size=bs,
                split_point=sp,
                client_offload_mb_num=bs,
                server_offload_mb_num=bs,
                client_offload_model_state_sp_num=sp,
                lora=lora,
            )
            sim_res = simulate(var, time_res, mem_res, save_gantt=False, save_time_res=True)
            all_data.append(
                {
                    'split_point': var.split_point,
                    'batch_size': var.batch_size,
                    **sim_res.objective.__dict__,
                }
            )
    df = pd.DataFrame(all_data)
    df = (
        df.where(df['client_peak_mem_alloc'] <= max_client_mem_mb * 0.95).sort_values(
            by=['server_cost_per_epoch', 'client_peak_mem_alloc'], ascending=[True, False]
        )
    ).round(2)
    # print(df[:10])
    df.to_csv(f'tmp_{model_name.split("/")[-1]}.csv', index=False)  # _fixed_batch
    # print(time_res)
    # do_optimize(model_name, dataset_size, max_split_point, max_batch_size, time_res, mem_res, max_client_mem_mb, lora)
    # do
    # real_data = pd.read_csv(f'log/extracted_optim/{model_name.split("/")[-1]}{f'_lora' if lora else '' }.csv')
    # # all_data = []
    # sim_client_mem = []
    # sim_server_mem = []
    # sim_batch_time = []
    # '''
    # batch_size,split_point,offload_mb_num,offload_ms_sp_num,client_mem,server_mem,batch_time
    # 8,1,8,1,7589.69,36199.55,7706.2
    # 16,1,16,1,7857.76,36339.55,14582.88
    # 24,1,24,1,8125.82,36479.55,21728.18
    # '''
    # for bs, sp, oam, osr in zip(
    #     real_data['batch_size'].values, real_data['split_point'].values, real_data['offload_mb_num'].values, real_data['offload_ms_sp_num'].values
    # ):
    #     # for bs in real_data['batch_size'].values:
    #     #     for sp in real_data['split_point'].values:
    #     #         # osr = sp
    #     #         # oam = bs
    #     #         for oam in real_data['offload_mb_num'].values:
    #     #             for osr in real_data['offload_ms_sp_num'].values:
    #     var = MainVariable(
    #         total_batch_num=1000,
    #         batch_size=bs,
    #         split_point=sp,
    #         client_offload_mb_num=oam,
    #         server_offload_mb_num=oam,
    #         client_offload_model_state_sp_num=osr,
    #         lora=lora,
    #     )
    #     sim_res = simulate(var, time_res, mem_res, save_gantt=False)
    #     sim_client_mem.append(sim_res.objective.client_peak_mem_alloc)
    #     sim_server_mem.append(sim_res.objective.server_peak_mem_alloc)
    #     sim_batch_time.append(sim_res.objective.batch_train_time)
    #     # all_data.append(
    #     #     {
    #     #         'batch_size': var.batch_size,
    #     #         'split_point': var.split_point,
    #     #         'offload_mb_num': var.client_offload_mb_num,
    #     #         'offload_ms_sp_num': var.client_offload_model_state_sp_num,
    #     #         'client_mem': round(sim_res.objective.client_peak_mem_alloc, 2),
    #     #         'server_mem': round(sim_res.objective.server_peak_mem_alloc, 2),
    #     #         'batch_time': round(sim_res.objective.batch_train_time, 2),
    #     #     }
    #     # )
    # # print(round(sim_res.objective.server_peak_mem_alloc, 2))
    # real_data['sim_client_mem'] = sim_client_mem
    # real_data['sim_server_mem'] = sim_server_mem
    # real_data['sim_batch_time'] = sim_batch_time
    # df = pd.DataFrame(real_data)
    # df = df.round(2)
    # df.to_csv(f'log/simulate_results_{model_name.split("/")[-1]}{'_lora' if lora else ''}.csv', index=False)
