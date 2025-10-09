from dataclasses import dataclass, field
from typing import List, Any, Dict
import numpy as np
import pandas as pd
import os
import json
import time
from usl.utils.usl_gantt_plot import plot_gantt_grouped


@dataclass
class MainVariable:
    """
    MainVariable contains the constant variables or user-defined variables of the system.
    """

    batch_size: int = 8
    micro_batch_size: int = 1
    max_seq_len: int = 512
    split_point: int = 4
    rate_mbps: float = 300  # mbps limit
    bytes_send_ps: float = 154339.0  # bytes per ms
    offload: bool = False
    lora: bool = False


@dataclass
class TimeVariable:
    """
    TimeVariable contains the time cost of different components in the system.
    """

    head_fwd_time: float = 17.5  # unit:ms
    head_bwd_time: float = 36.77
    server_fwd_time: float = 31.61
    server_bwd_time: float = 66.1
    head_activation_send_time: float = 142  # unit:ms
    head_gradient_send_time: float = 109
    # most of the time,server_activation_recv_time approximates head_gradient_recv_time and server gradient send time
    server_activation_send_time: float = 108
    server_gradient_send_time: float = 107  # unit:ms
    tail_fwd_time: float = 29.09
    tail_bwd_time: float = 66.67  # unit:ms


@dataclass
class MemoryVariable:
    """
    MemoryVariable contains the memory usage of different components in the system.
    """

    non_block_param_mem: float = 0  # unit:MB
    non_block_fwd_activation_mem: float = 0
    non_block_fwd_peak_mem: float = 0
    block_param_mem: float = 0
    block_fwd_activation_mem: float = 0
    block_fwd_peak_mem: float = 0


@dataclass
class Objective:
    """
    Objective contains the objective function values of different components in the system,used to do the optimization.
    """

    client_idle_rate: float = 0.0  # unit:%
    server_idle_rate: float = 0.0
    client_send_rate: float = 0.0
    server_send_rate: float = 0.0
    client_peak_mem_alloc: float = 0.0  # unit:MB
    server_peak_mem_alloc: float = 0.0
    batch_train_time: float = 0.0  # unit:ms
    pass


@dataclass
class SimulateResult:
    """
    SimulateResult contains the result of the simulation.
    """

    main_variable: MainVariable = field(default_factory=lambda: MainVariable())
    time_variable: TimeVariable = field(default_factory=lambda: TimeVariable())
    memory_variable: MemoryVariable = field(default_factory=lambda: MemoryVariable())
    objective: Objective = field(default_factory=lambda: Objective())

    def to_dict(self):
        return {
            "constant": self.main_variable.__dict__,
            "time_variable": self.time_variable.__dict__,
            "memory_variable": self.memory_variable.__dict__,
            "objective": self.objective.__dict__,
        }

    def save_to_json(self, file_path='default_result.json'):
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


def _simulate_train_time(
    main_var: MainVariable, time_var: TimeVariable, memory_var: MemoryVariable, objective: Objective, save_gantt: bool = False
) -> SimulateResult:
    # simulate the batch training time
    micro_batch_num = (main_var.batch_size + main_var.micro_batch_size - 1) // main_var.micro_batch_size
    # use list to do scheduling, each element is a list of two elements, [start_time, end_time]
    head_fwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    head_bwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_fwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
    server_bwd_timestamps = [[0, 0] for _ in range(micro_batch_num)]
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
            head_fwd_timestamps[0][1] = head_fwd_timestamps[0][0] + time_var.head_fwd_time
        else:
            head_fwd_timestamps[i][0] = head_fwd_timestamps[i - 1][1]
            head_fwd_timestamps[i][1] = head_fwd_timestamps[i][0] + time_var.head_fwd_time
    # step2 : do head activation send
    for i in range(micro_batch_num):
        if i == 0:
            head_activation_send_timestamps[0][0] = head_fwd_timestamps[0][1]
        else:
            head_activation_send_timestamps[i][0] = max(head_fwd_timestamps[i][1], head_activation_send_timestamps[i - 1][1])
        head_activation_send_timestamps[i][1] = head_activation_send_timestamps[i][0] + time_var.head_activation_send_time
    # step3 : do server fwd
    for i in range(micro_batch_num):
        if i == 0:
            server_fwd_timestamps[0][0] = head_activation_send_timestamps[0][1]
        else:
            server_fwd_timestamps[i][0] = max(head_activation_send_timestamps[i][1], server_fwd_timestamps[i - 1][1])
        server_fwd_timestamps[i][1] = server_fwd_timestamps[i][0] + time_var.server_fwd_time
    # step4 : do server activation send
    for i in range(micro_batch_num):
        if i == 0:
            server_activation_send_timestamps[0][0] = server_fwd_timestamps[0][1]
        else:
            server_activation_send_timestamps[i][0] = max(server_fwd_timestamps[i][1], server_activation_send_timestamps[i - 1][1])
        server_activation_send_timestamps[i][1] = server_activation_send_timestamps[i][0] + time_var.server_activation_send_time
    # step5 : do tail fwd and bwd
    for i in range(micro_batch_num):
        if i == 0:
            tail_fwd_timestamps[0][0] = max(head_fwd_timestamps[-1][1], server_activation_send_timestamps[0][1])
        else:
            tail_fwd_timestamps[i][0] = max(tail_fwd_timestamps[i - 1][1], server_activation_send_timestamps[i][1])
        tail_fwd_timestamps[i][1] = tail_fwd_timestamps[i][0] + time_var.tail_fwd_time
        tail_bwd_timestamps[i][0] = tail_fwd_timestamps[i][1]
        tail_bwd_timestamps[i][1] = tail_bwd_timestamps[i][0] + time_var.tail_bwd_time

    # step6 : do client grad send to server
    for i in range(micro_batch_num):
        if i == 0:
            tail_gradient_send_timestamps[0][0] = max(head_activation_send_timestamps[-1][1], tail_bwd_timestamps[0][1])
        else:
            tail_gradient_send_timestamps[i][0] = max(tail_gradient_send_timestamps[i - 1][1], tail_bwd_timestamps[i][1])
        tail_gradient_send_timestamps[i][1] = tail_gradient_send_timestamps[i][0] + time_var.head_gradient_send_time

    # step7 : do server bwd
    for i in range(micro_batch_num):
        if i == 0:
            server_bwd_timestamps[0][0] = max(server_fwd_timestamps[-1][1], tail_gradient_send_timestamps[0][1])
        else:
            server_bwd_timestamps[i][0] = max(server_bwd_timestamps[i - 1][1], tail_gradient_send_timestamps[i][1])
        server_bwd_timestamps[i][1] = server_bwd_timestamps[i][0] + time_var.server_bwd_time
    # step8 : do server grad send to head
    for i in range(micro_batch_num):
        if i == 0:
            server_gradient_send_timestamps[0][0] = max(server_activation_send_timestamps[-1][1], server_bwd_timestamps[0][1])
        else:
            server_gradient_send_timestamps[i][0] = max(server_gradient_send_timestamps[i - 1][1], server_bwd_timestamps[i][1])
        server_gradient_send_timestamps[i][1] = server_gradient_send_timestamps[i][0] + time_var.server_gradient_send_time

    # step9 : do head bwd
    for i in range(micro_batch_num):
        if i == 0:
            head_bwd_timestamps[0][0] = max(tail_bwd_timestamps[-1][1], server_gradient_send_timestamps[0][1])
        else:
            head_bwd_timestamps[i][0] = max(head_bwd_timestamps[i - 1][1], server_gradient_send_timestamps[i][1])
        head_bwd_timestamps[i][1] = head_bwd_timestamps[i][0] + time_var.head_bwd_time
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
        plot_gantt_grouped(gantt_data, fp='log/img/grouped/simulated_gantt.png', align=False)
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
        main_var,
        time_var,
        memory_var,
        Objective(
            client_idle_rate=client_idle_rate,
            server_idle_rate=server_idle_rate,
            client_send_rate=client_send_rate,
            server_send_rate=server_send_rate,
            client_peak_mem_alloc=0,
            server_peak_mem_alloc=0,
            batch_train_time=batch_train_time,
        ),
    )


def _simulate_peak_mem_alloc(
    main_variable: MainVariable, time_variable: TimeVariable, memory_variable: MemoryVariable, objective: Objective
) -> SimulateResult:
    # simulate the peak memory allocation
    pass


def simulate(main_variable: MainVariable, time_variable: TimeVariable, memory_variable: MemoryVariable, objective: Objective) -> SimulateResult:
    simulate_result = _simulate_train_time(main_variable, time_variable, memory_variable, objective, True)
    # simulate_mem_result = _simulate_peak_mem_alloc(main_variable, time_variable, memory_variable, objective)
    # copy memory result to simulate_result
    # simulate_result.objective.client_peak_mem_alloc = simulate_mem_result.objective.client_peak_mem_alloc
    # simulate_result.objective.server_peak_mem_alloc = simulate_mem_result.objective.server_peak_mem_alloc
    return simulate_result


simulate_res = simulate(MainVariable(), TimeVariable(), MemoryVariable(), Objective())
print(simulate_res.to_dict())
