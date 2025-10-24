import subprocess
import json
import os
import argparse
from typing import Tuple
from usl.simulate import MemoryConstant, TimeConstant
from transformers import AutoConfig

OFFLOAD_VALUES = ["", "-OA", "-OS"]


def _get_layer_num(dir='data/models', model_name='gpt2-large') -> AutoConfig:
    fp = os.path.join(dir, model_name, 'config.json')
    try:
        config = AutoConfig.from_pretrained(fp)
    except:
        raise ValueError(f"Failed to load config from {fp}")
    if hasattr(config, 'num_hidden_layers'):
        return config.num_hidden_layers
    elif hasattr(config, 'n_layer'):
        return config.n_layer
    elif hasattr(config, 'num_layers'):
        return config.num_layers
    else:
        raise ValueError(f"Failed to get layer number from config {config}")


def run_profile(
    model: str, mbps: float, lora: bool, base_bs=4, base_sp=2, profile_dir: str = 'log/profile/sim_profile'
) -> Tuple[MemoryConstant, TimeConstant]:
    """
    _summary_

    Args:
        model (str): _description_
        batch_size (int): _description_
        mbps (float): _description_
        lora (bool): _description_
        profile_dir (str, optional): used to save profile data during simulation. Defaults to 'log/profile/sim_profile'.

    Returns:
        Tuple[MemoryConstant, TimeConstant]: _description_
    """
    try:
        layer_num = _get_layer_num(model_name=model)
    except ValueError as e:
        print(e)
        return None, None
    max_split_point = layer_num // 2
    # batch_sizes = [base_bs, base_bs + 1]
    split_points = [base_sp, base_sp + 1]
    prof_res = {}
    print('Memory profile finished.')
    # Iterate over split_points and offload_values dynamically
    # for sp in split_points:
    #     prof_res[sp] = {}
    #     for offload in OFFLOAD_VALUES:
    #         prof_res[sp][offload] = {}
    #         # Generate the file name dynamically based on the parameters
    #         offload_str = f"_coa_{base_bs}_soa_{base_bs}" if offload == '-OA' else f"_cos_{sp}" if offload == '-OS' else ""
    #         file_name = f"sp_{sp}_b_{base_bs}_mb_1_s_512_mbps_{mbps}_pipedream_wc{'_lora' if lora else '' }{offload_str}.json"
    #         file_path = os.path.join(profile_dir, model, file_name)
    #         # Read and process the file
    #         try:
    #             with open(file_path, 'r') as f:
    #                 data: dict = json.load(f)
    #             # Store the parsed data
    #             prof_res[sp][offload] = data
    #         except FileNotFoundError:
    #             print(f"File {file_path} not found. Skipping.")
    #             continue
    combinations = [
        # sp,coa,cos,soa
        (base_sp, 0, 0, 0),  # 什么都不做
        (base_sp, base_bs, 0, base_bs),  # 卸载激活量
        (base_sp + 1, 0, 0, 0),
        (base_sp + 1, base_bs, 0, base_bs),  # 卸载激活量
        (base_sp, 0, base_sp, 0),  # 卸载一层模型参数
        (base_sp + 1, 0, base_sp + 1, 0),  # 卸载两层模型参数
    ]

    def get_file_name(sp, coa, cos, soa):
        offload_str = f"_coa_{coa}" if coa > 0 else ""
        offload_str += f"_cos_{cos}" if cos > 0 else ""
        offload_str += f"_soa_{soa}" if soa > 0 else ""
        file_path = (
            f"log/profile/sim_profile/{model}/sp_{sp}_b_{base_bs}_mb_1_s_512_mbps_{mbps}_pipedream_wc{'_lora' if lora else '' }{offload_str}.json"
        )
        return file_path

    for comb in combinations:
        sp, coa, cos, soa = comb
        file_path = get_file_name(sp, coa, cos, soa)
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Skipping.")
            continue
        try:
            with open(file_path, 'r') as f:
                data: dict = json.load(f)
            # Store the parsed data
            prof_res[comb] = data
        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping.")
            continue

    # MemoryVariable calculations
    mem_var = MemoryConstant(max_split_point=max_split_point, baseline_split_point=base_sp, baseline_minibatch_num=base_bs)
    mem_var.mem_increment_per_sp_server = round(
        prof_res[(base_sp + 1, 0, 0, 0)]['server_max_mem_alloc_mb'] - prof_res[(base_sp, 0, 0, 0)]['server_max_mem_alloc_mb'], 2
    )  # checked
    mem_var.mem_increment_per_sp_client = round(
        prof_res[(base_sp + 1, 0, 0, 0)]['client_max_mem_alloc_mb'] - prof_res[(base_sp, 0, 0, 0)]['client_max_mem_alloc_mb'], 2
    )  # checked
    mem_var.mem_increment_per_sp_mb_client = round(
        (prof_res[(base_sp, 0, 0, 0)]['client_max_mem_alloc_mb'] - prof_res[(base_sp, base_bs, 0, base_bs)]['client_max_mem_alloc_mb'])
        / (base_bs - 1)
        / base_sp,
        2,
    )  # checked
    mem_var.mem_increment_per_sp_mb_server = round(
        (prof_res[(base_sp, 0, 0, 0)]['server_max_mem_alloc_mb'] - prof_res[(base_sp, base_bs, 0, base_bs)]['server_max_mem_alloc_mb'])
        / (base_bs - 3)
        / base_sp,
        2,
    )  # checked
    mem_var.base_client_mem_alloc = round(prof_res[(base_sp, 0, 0, 0)]['client_max_mem_alloc_mb'], 2)
    mem_var.base_server_mem_alloc = (
        round(prof_res[(base_sp, 0, 0, 0)]['server_max_mem_alloc_mb'], 2) + (max_split_point - base_sp * 2) * mem_var.mem_increment_per_sp_server
    )  # checked
    mem_var.base_model_state_mem_alloc_client = round(
        prof_res[(base_sp, 0, 0, 0)]['client_max_mem_alloc_mb'] - prof_res[(base_sp, 0, base_sp, 0)]['client_max_mem_alloc_mb'], 2
    )  # checked
    mem_var.model_mem_increment_per_sp_client = (
        prof_res[(base_sp + 1, 0, 0, 0)]['client_max_mem_alloc_mb']
        - prof_res[(base_sp + 1, 0, base_sp + 1, 0)]['client_max_mem_alloc_mb']
        - (prof_res[(base_sp, 0, 0, 0)]['client_max_mem_alloc_mb'] - prof_res[(base_sp, 0, base_sp, 0)]['client_max_mem_alloc_mb'])
    )  # checked ,but some loss
    mem_var.base_model_state_mem_alloc_except_blocks = mem_var.base_model_state_mem_alloc_client - base_sp * mem_var.model_mem_increment_per_sp_client
    # TimeVariable calculations
    time_var = TimeConstant(rate_mbps=mbps)
    # base time
    time_var.base_head_fwd_time_per_mb = prof_res[(base_sp, 0, 0, 0)]['head_fwd_time_avg_ms']
    time_var.base_head_bwd_time_per_mb = prof_res[(base_sp, 0, 0, 0)]['head_bwd_time_avg_ms']
    time_var.base_tail_fwd_time_per_mb = prof_res[(base_sp, 0, 0, 0)]['tail_fwd_time_avg_ms']
    time_var.base_tail_bwd_time_per_mb = prof_res[(base_sp, 0, 0, 0)]['tail_bwd_time_avg_ms']
    time_var.base_server_fwd_time_per_mb = prof_res[(base_sp, 0, 0, 0)]['server_fwd_time_avg_ms']
    time_var.base_server_bwd_time_per_mb = prof_res[(base_sp, 0, 0, 0)]['server_bwd_time_avg_ms']
    # increment per sp
    time_var.head_fwd_time_increment_per_sp = (
        prof_res[(base_sp + 1, 0, 0, 0)]['head_fwd_time_avg_ms'] - prof_res[(base_sp, 0, 0, 0)]['head_fwd_time_avg_ms']
    )
    time_var.head_bwd_time_increment_per_sp = (
        prof_res[(base_sp + 1, 0, 0, 0)]['head_bwd_time_avg_ms'] - prof_res[(base_sp, 0, 0, 0)]['head_bwd_time_avg_ms']
    )
    time_var.tail_fwd_time_increment_per_sp = (
        prof_res[(base_sp + 1, 0, 0, 0)]['tail_fwd_time_avg_ms'] - prof_res[(base_sp, 0, 0, 0)]['tail_fwd_time_avg_ms']
    )
    time_var.tail_bwd_time_increment_per_sp = (
        prof_res[(base_sp + 1, 0, 0, 0)]['tail_bwd_time_avg_ms'] - prof_res[(base_sp, 0, 0, 0)]['tail_bwd_time_avg_ms']
    )
    time_var.server_fwd_time_increment_per_sp = (
        prof_res[(base_sp + 1, 0, 0, 0)]['server_fwd_time_avg_ms'] - prof_res[(base_sp, 0, 0, 0)]['server_fwd_time_avg_ms']
    )
    time_var.server_bwd_time_increment_per_sp = (
        prof_res[(base_sp + 1, 0, 0, 0)]['server_bwd_time_avg_ms'] - prof_res[(base_sp, 0, 0, 0)]['server_bwd_time_avg_ms']
    )
    # model state offload time
    # "head_m_offload_time_ms": 10.86,
    # "head_m_reload_time_ms": 10.86,
    # "tail_m_offload_time_ms": 10.85,
    # "tail_m_reload_time_ms": 10.87,
    # "head_os_offload_time_ms": 114.92,
    # "head_os_reload_time_ms": 107.88,
    # "tail_os_offload_time_ms": 170.07,
    # "tail_os_reload_time_ms": 115.47,
    # "activation_offload_time_ms": 0,
    # "activation_reload_time_ms": 0,
    time_var.base_head_model_state_offload_time = (
        prof_res[(base_sp, 0, base_sp, 0)]['head_m_offload_time_ms'] + prof_res[(base_sp, 0, base_sp, 0)]['head_os_offload_time_ms']
    )
    time_var.base_tail_model_state_offload_time = min(
        prof_res[(base_sp, 0, base_sp, 0)]['tail_m_offload_time_ms'] + prof_res[(base_sp, 0, base_sp, 0)]['tail_os_offload_time_ms'],
        time_var.base_head_model_state_offload_time,
    )
    # time_var.head_model_offload_time_increment_per_sp = (
    #     prof_res[(base_sp + 1, 0, base_sp + 1, 0)]['head_m_offload_time_ms']
    #     + prof_res[(base_sp + 1, 0, base_sp + 1, 0)]['head_os_offload_time_ms']
    #     - (prof_res[(base_sp, 0, base_sp, 0)]['head_m_offload_time_ms'] + prof_res[(base_sp, 0, base_sp, 0)]['head_os_offload_time_ms'])
    # )
    time_var.head_model_offload_time_increment_per_sp = prof_res[(base_sp, 0, base_sp, 0)]['head_m_offload_time_ms'] / base_sp * 3  # param+os
    time_var.tail_model_offload_time_increment_per_sp = time_var.head_model_offload_time_increment_per_sp
    # head activation offload time
    base_activation_offload_time_ms = prof_res[(base_sp, base_bs, 0, base_bs)]['activation_offload_time_ms'][:-1]
    base_1_activation_offload_time_ms = prof_res[(base_sp + 1, base_bs, 0, base_bs)]['activation_offload_time_ms'][:-1]
    time_var.base_head_activation_offload_time_per_mb = sum(base_activation_offload_time_ms) / len(base_activation_offload_time_ms)
    time_var.head_activation_offload_time_increment_per_sp = sum(base_1_activation_offload_time_ms) / len(base_1_activation_offload_time_ms) - sum(
        base_activation_offload_time_ms
    ) / len(base_activation_offload_time_ms)
    base_activation_reload_time_ms = prof_res[(base_sp, base_bs, 0, base_bs)]['activation_reload_time_ms'][:-1]
    base_1_activation_reload_time_ms = prof_res[(base_sp + 1, base_bs, 0, base_bs)]['activation_reload_time_ms'][:-1]
    time_var.base_head_activation_reload_time_per_mb = sum(base_activation_reload_time_ms) / len(base_activation_reload_time_ms)
    time_var.head_activation_reload_time_increment_per_sp = sum(base_1_activation_reload_time_ms) / len(base_1_activation_reload_time_ms) - sum(
        base_activation_reload_time_ms
    ) / len(base_activation_reload_time_ms)

    # delay time
    time_var.delay_time_avg_ms = (
        prof_res[(base_sp, 0, 0, 0)]['delay_time_avg_ms'] + prof_res[(base_sp, base_bs, 0, base_bs)]['delay_time_avg_ms']
    ) / 2
    # network time
    time_var.head_activation_send_time = (
        prof_res[(base_sp, 0, 0, 0)]['head_fwd_send_time_avg_ms'] + prof_res[(base_sp, base_bs, 0, base_bs)]['head_fwd_send_time_avg_ms']
    ) / 2
    time_var.server_activation_send_time = (
        prof_res[(base_sp, 0, 0, 0)]['server_fwd_send_time_avg_ms'] + prof_res[(base_sp, base_bs, 0, base_bs)]['server_fwd_send_time_avg_ms']
    ) / 2
    time_var.tail_gradient_send_time = (
        prof_res[(base_sp, 0, 0, 0)]['tail_bwd_send_time_avg_ms'] + prof_res[(base_sp, base_bs, 0, base_bs)]['tail_bwd_send_time_avg_ms']
    ) / 2
    time_var.server_gradient_send_time = (
        prof_res[(base_sp, 0, 0, 0)]['server_bwd_send_time_avg_ms'] + prof_res[(base_sp, base_bs, 0, base_bs)]['server_bwd_send_time_avg_ms']
    ) / 2
    # round all values to 2 decimal places
    keys = time_var.__dict__.keys()
    for key in keys:
        time_var.__dict__[key] = round(time_var.__dict__[key], 2)
        if time_var.__dict__[key] < 0:
            print(f'warning: negative time value for key: {key},please check the profiling results.')

    return mem_var, time_var
