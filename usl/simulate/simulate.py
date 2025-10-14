import argparse

import math
import os
import json
import time
from usl.utils.usl_gantt_plot import plot_gantt_grouped
from usl.simulate import *
from usl.simulate.profile import run_profile


def _simulate_train_time(main_var: MainVariable, time_const: TimeConstant, mem_const: MemoryConstant, save_gantt: bool = False) -> SimulateResult:
    # simulate the batch training time
    micro_batch_num = (main_var.batch_size + mem_const.micro_batch_size - 1) // mem_const.micro_batch_size
    split_point = main_var.split_point
    assert split_point > 0, "split_point should be greater than 0"
    if not main_var.offload:
        head_fwd_time = time_const.base_no_off_head_fwd_time + (split_point - 1) * time_const.head_no_off_fwd_time_increment_per_sp
        head_bwd_time = time_const.base_no_off_head_bwd_time + (split_point - 1) * time_const.head_no_off_bwd_time_increment_per_sp
        head_offload_time = 0
        head_reload_time = 0
        tail_reload_time = 0
        tail_offload_time = 0
        server_fwd_time = time_const.base_no_off_server_fwd_time - (split_point - 1) * time_const.server_no_off_fwd_time_increment_per_sp
        server_bwd_time = time_const.base_no_off_server_bwd_time - (split_point - 1) * time_const.server_no_off_bwd_time_increment_per_sp
        tail_fwd_time = time_const.base_tail_fwd_time + (split_point - 1) * time_const.tail_fwd_time_increment_per_sp
        tail_bwd_time = time_const.base_tail_bwd_time + (split_point - 1) * time_const.tail_bwd_time_increment_per_sp
    else:
        head_fwd_time = time_const.base_off_head_fwd_time + (split_point - 1) * time_const.head_off_fwd_time_increment_per_sp
        head_bwd_time = time_const.base_off_head_bwd_time + (split_point - 1) * time_const.head_off_bwd_time_increment_per_sp
        head_offload_time = time_const.base_head_offload_time + (split_point - 1) * time_const.head_offload_time_increment_per_sp
        head_reload_time = head_offload_time  # assume that offload = reload
        tail_reload_time = time_const.base_tail_offload_time + (split_point - 1) * time_const.tail_offload_time_increment_per_sp
        tail_offload_time = tail_reload_time
        server_fwd_time = time_const.base_off_server_fwd_time - (split_point - 1) * time_const.server_off_fwd_time_increment_per_sp
        server_bwd_time = time_const.base_off_server_bwd_time - (split_point - 1) * time_const.server_off_bwd_time_increment_per_sp
        tail_fwd_time = time_const.base_tail_fwd_time + (split_point - 1) * time_const.tail_fwd_time_increment_per_sp
        tail_bwd_time = time_const.base_tail_bwd_time + (split_point - 1) * time_const.tail_bwd_time_increment_per_sp
    print('time: ', offload, split_point, server_fwd_time, server_bwd_time)
    # print(head_offload_time, tail_reload_time)
    # use list to do scheduling, each element is a list of two elements, [start_time, end_time]
    head_fwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    head_offload_timestamp = [0, 0]
    tail_reload_timestamp = [0, 0]
    head_bwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_fwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_bwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    tail_offload_timestamp = [0, 0]
    head_reload_timestamp = [0, 0]
    tail_fwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    tail_bwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    head_activation_send_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    tail_gradient_send_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_activation_send_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_gradient_send_timestamps = [[0, 0] for _ in range(micro_batch_num)]

    # do simulating
    # step1 : do head fwd
    for i in range(micro_batch_num):
        if i == 0:
            head_fwd_timestamps[0][1] = head_fwd_timestamps[0][0] + head_fwd_time
        else:
            head_fwd_timestamps[i][0] = head_fwd_timestamps[i - 1][1]
            head_fwd_timestamps[i][1] = head_fwd_timestamps[i][0] + head_fwd_time
    # step 1.1 do offload if needed
    if main_var.offload:
        head_offload_timestamp[0] = head_fwd_timestamps[-1][1]
        head_offload_timestamp[1] = head_offload_timestamp[0] + head_offload_time
        tail_reload_timestamp[0] = head_fwd_timestamps[-1][1]
        tail_offload_timestamp[1] = tail_reload_timestamp[0] + tail_reload_time
    # step2 : do head activation send
    for i in range(micro_batch_num):
        if i == 0:
            head_activation_send_timestamps[0][0] = head_fwd_timestamps[0][1]
        else:
            head_activation_send_timestamps[i][0] = max(head_fwd_timestamps[i][1], head_activation_send_timestamps[i - 1][1])
        head_activation_send_timestamps[i][1] = head_activation_send_timestamps[i][0] + time_const.head_activation_send_time
    # step3 : do server fwd
    for i in range(micro_batch_num):
        if i == 0:
            server_fwd_timestamps[0][0] = head_activation_send_timestamps[0][1]
        else:
            server_fwd_timestamps[i][0] = max(head_activation_send_timestamps[i][1], server_fwd_timestamps[i - 1][1])
        server_fwd_timestamps[i][1] = server_fwd_timestamps[i][0] + server_fwd_time
    # step4 : do server activation send
    for i in range(micro_batch_num):
        if i == 0:
            server_activation_send_timestamps[0][0] = server_fwd_timestamps[0][1]
        else:
            server_activation_send_timestamps[i][0] = max(server_fwd_timestamps[i][1], server_activation_send_timestamps[i - 1][1])
        server_activation_send_timestamps[i][1] = server_activation_send_timestamps[i][0] + time_const.server_activation_send_time
    # step5 : do tail fwd and bwd
    for i in range(micro_batch_num):
        if i == 0:
            tail_fwd_timestamps[0][0] = max(
                head_fwd_timestamps[-1][1] + head_offload_time, head_fwd_timestamps[-1][1] + tail_reload_time, server_activation_send_timestamps[0][1]
            )
        else:
            tail_fwd_timestamps[i][0] = max(tail_bwd_timestamps[i - 1][1], server_activation_send_timestamps[i][1])
        tail_fwd_timestamps[i][1] = tail_fwd_timestamps[i][0] + tail_fwd_time
        tail_bwd_timestamps[i][0] = tail_fwd_timestamps[i][1]
        tail_bwd_timestamps[i][1] = tail_bwd_timestamps[i][0] + tail_bwd_time

    # step6 : do client grad send to server
    for i in range(micro_batch_num):
        if i == 0:
            tail_gradient_send_timestamps[0][0] = max(head_activation_send_timestamps[-1][1], tail_bwd_timestamps[0][1])
        else:
            tail_gradient_send_timestamps[i][0] = max(tail_gradient_send_timestamps[i - 1][1], tail_bwd_timestamps[i][1])
        tail_gradient_send_timestamps[i][1] = tail_gradient_send_timestamps[i][0] + time_const.tail_gradient_send_time

    # step7 : do server bwd
    for i in range(micro_batch_num):
        if i == 0:
            server_bwd_timestamps[0][0] = max(server_fwd_timestamps[-1][1], tail_gradient_send_timestamps[0][1])
        else:
            server_bwd_timestamps[i][0] = max(server_bwd_timestamps[i - 1][1], tail_gradient_send_timestamps[i][1])
        server_bwd_timestamps[i][1] = server_bwd_timestamps[i][0] + server_bwd_time
    # step8 : do server grad send to head
    for i in range(micro_batch_num):
        if i == 0:
            server_gradient_send_timestamps[0][0] = max(server_activation_send_timestamps[-1][1], server_bwd_timestamps[0][1])
        else:
            server_gradient_send_timestamps[i][0] = max(server_gradient_send_timestamps[i - 1][1], server_bwd_timestamps[i][1])
        server_gradient_send_timestamps[i][1] = server_gradient_send_timestamps[i][0] + time_const.server_gradient_send_time

    if main_var.offload:
        head_reload_timestamp[0] = head_fwd_timestamps[-1][1]
        head_reload_timestamp[1] = head_offload_timestamp[0] + head_reload_time
        tail_offload_timestamp[0] = head_fwd_timestamps[-1][1]
        tail_offload_timestamp[1] = tail_offload_timestamp[0] + tail_offload_time
    # step9 : do head bwd
    for i in range(micro_batch_num):
        if i == 0:
            head_bwd_timestamps[0][0] = max(
                tail_bwd_timestamps[-1][1] + tail_offload_time, tail_bwd_timestamps[-1][1] + head_reload_time, server_gradient_send_timestamps[0][1]
            )
        else:
            head_bwd_timestamps[i][0] = max(head_bwd_timestamps[i - 1][1], server_gradient_send_timestamps[i][1])
        head_bwd_timestamps[i][1] = head_bwd_timestamps[i][0] + head_bwd_time
    # print(head_fwd_timestamps)
    if save_gantt:
        gantt_data = [
            {
                "mini_batch_idx": i,
                "train_time_duration_ms": head_bwd_timestamps[i][1] - head_fwd_timestamps[i][0],
                "head_fwd_timestamp": head_fwd_timestamps[i],
                "head_fwd_send_timestamp": head_activation_send_timestamps[i],
                "server_fwd_timestamp": server_fwd_timestamps[i],
                "server_fwd_send_timestamp": server_activation_send_timestamps[i],
                "tail_fwd_timestamp": tail_fwd_timestamps[i],
                "tail_bwd_timestamp": tail_bwd_timestamps[i],
                "tail_bwd_send_timestamp": tail_gradient_send_timestamps[i],
                "server_bwd_timestamp": server_bwd_timestamps[i],
                "server_bwd_send_timestamp": server_gradient_send_timestamps[i],
                "head_bwd_timestamp": head_bwd_timestamps[i],
            }
            for i in range(micro_batch_num)
        ]
        save_dir = f'log/img/simulated/{model_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fp = f'{save_dir}/sp_{split_point}_b_{max_batch_size}_mb_{mem_const.micro_batch_size}_s_{mem_const.max_seq_len}_mbps_{mbps}_pipedream_wc{"_lora" if lora else ""}{"_oa_os" if main_var.offload else ""}.png'
        plot_gantt_grouped(gantt_data, fp, align=False)
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
    for i in range(micro_batch_num):
        server_compute_time += server_fwd_timestamps[i][1] - server_fwd_timestamps[i][0] + server_bwd_timestamps[i][1] - server_bwd_timestamps[i][0]
        server_idle_rate = 100 - (server_compute_time / batch_train_time) * 100
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
    return SimulateResult(
        main_variable=main_var,
        time_const=time_const,
        objective=Objective(
            client_idle_rate=round(client_idle_rate, 2),
            server_idle_rate=round(server_idle_rate, 2),
            client_send_rate=round(client_send_rate, 2),
            server_send_rate=round(server_send_rate, 2),
            client_peak_mem_alloc=0,
            server_peak_mem_alloc=0,
            batch_train_time=round(batch_train_time, 2),
        ),
    )


def _simulate_peak_mem_alloc(main_var: MainVariable, memory_const: MemoryConstant) -> SimulateResult:
    # simulate the peak memory allocation
    split_point = main_var.split_point
    batch_size = main_var.batch_size
    max_split_point=memory_const.max_split_point
    assert split_point >= 1, "split_point should be greater than or equal to 1"
    assert batch_size >= 4, "batch size should be greater than or equal to 4"
    if main_var.offload:
        client_peak_mem_alloc = (
            memory_const.base_max_mem_alloc_off_client
            + (split_point - 1) * memory_const.offload_mem_increment_per_sp_client
            + (batch_size - 4) * memory_const.offload_mem_increment_per_mb_client
        )
        server_max_mem_alloc = (
            memory_const.base_max_mem_alloc_off_server
            - (split_point - 1) * memory_const.offload_mem_decrement_per_sp_server
            + (max_split_point-split_point) * (batch_size - 4) * memory_const.offload_mem_increment_per_mb_server
        )
    else:
        client_peak_mem_alloc = (
            memory_const.base_max_mem_alloc_no_off_client
            + (split_point - 1) * memory_const.no_off_mem_increment_per_sp_client
            + (split_point) * (batch_size - 4) * memory_const.no_off_mem_increment_per_mb_client
        )
        server_max_mem_alloc = (
            memory_const.base_max_mem_alloc_no_off_server
            - (split_point - 1) * memory_const.no_off_mem_decrement_per_sp_server
            + (max_split_point-split_point) * (batch_size - 4) * memory_const.no_off_mem_increment_per_mb_server
            + (batch_size - 4) * memory_const.offload_mem_increment_per_mb_client
        )

    return SimulateResult(objective=Objective(client_peak_mem_alloc=client_peak_mem_alloc, server_peak_mem_alloc=server_max_mem_alloc))


def simulate(main_var: MainVariable, time_const: TimeConstant, mem_const: MemoryConstant, save_gantt: bool = False) -> SimulateResult:
    simulate_mem_result = _simulate_peak_mem_alloc(main_var, mem_const)
    simulate_result = _simulate_train_time(main_var, time_const, mem_const, save_gantt)
    # copy memory result to simulate_result
    simulate_result.objective.client_peak_mem_alloc = simulate_mem_result.objective.client_peak_mem_alloc
    simulate_result.objective.server_peak_mem_alloc = simulate_mem_result.objective.server_peak_mem_alloc
    print(simulate_mem_result.objective.server_peak_mem_alloc)
    return simulate_result


def parse_arguments():
    parser = argparse.ArgumentParser(description='Simulate memory and time profiling with dynamic parameters.')

    # Defining the command-line arguments
    parser.add_argument('--model', type=str, default='meta-llama/llama3.2-1b', help='The model name.')
    parser.add_argument('--max_client_mem_gb', type=int, default=8, help='The maximum memory allocation for the client.')
    parser.add_argument('--max_split_point', '-MSP', type=int, default=4, help='The number of layers in the model.')
    parser.add_argument('--lora', action='store_true', help='Whether to use Lora or not.')
    parser.add_argument('--mbps', type=int, default=1000, help='The mbps value for the simulation.')
    parser.add_argument('--max_batch_size', '-BS', type=int, default=64, help='The max batch size for the simulation.')
    parser.add_argument('--profile_dir', type=str, default='log/sim_profile', help='The profile directory for storing results.')
    parser.add_argument('--save_gantt', action='store_true', help='Whether to save gantt chart or not.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_name = args.model
    lora = args.lora
    mbps = args.mbps
    max_batch_size = args.max_batch_size
    profile_dir = args.profile_dir
    max_split_point = args.max_split_point
    max_client_mem_mb = args.max_client_mem_gb * 1024
    save_gantt = args.save_gantt
    # run profiling
    mem_res, time_res = run_profile(model_name, mbps, lora, profile_dir)
    if mem_res is None or time_res is None:
        print("Failed to run profiling.")
        exit(1)
    # create main variable
    best_strategy = None
    # batch_sizes=[2**i for i in range(2,math.ceil(math.log2(max_batch_size+1)))]
    for bs in [32,64]:
        for sp in range(1, max_split_point + 1):
            for offload in [True, False]:
                var = MainVariable(batch_size=bs, split_point=sp, offload=offload, lora=lora)
                # run simulation
                sim_res = simulate(var, time_res, mem_res, save_gantt=False)
                sim_res.model_name = model_name
                if best_strategy is None:
                    best_strategy = sim_res
                else:
                    time_diff = abs(sim_res.objective.batch_train_time - best_strategy.objective.batch_train_time)
                    mem_diff = abs(sim_res.objective.client_peak_mem_alloc - best_strategy.objective.client_peak_mem_alloc)
                    if (
                        mem_diff / best_strategy.objective.client_peak_mem_alloc > time_diff / best_strategy.objective.batch_train_time
                        and sim_res.objective.client_peak_mem_alloc < max_client_mem_mb
                    ):
                        # print(best_strategy)
                        best_strategy = sim_res
                save_dir = f'log/profile/simulate/{model_name}'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                sim_res.save_to_json(
                    f'{save_dir}/sp_{sp}_b_{bs}_mb_{mem_res.micro_batch_size}_s_{mem_res.max_seq_len}_mbps_{mbps}_pipedream_wc{"_lora"if lora else ""}{'_oa_os' if offload else ''}.json'
                )
    print(best_strategy.main_variable)
