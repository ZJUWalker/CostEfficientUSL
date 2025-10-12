import subprocess
import json
import os
import argparse
from typing import Tuple
from usl.simulate import MemoryConstant, TimeConstant
from transformers import AutoConfig


SPLIT_POINTS = [1, 2]
OFFLOAD_VALUES = [False, True]


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


def run_profile(model: str, batch_size: int, mbps: float, lora: bool, profile_dir: str = 'log/sim_profile') -> Tuple[MemoryConstant, TimeConstant]:
    """
    _summary_

    Args:
        model (str): _description_
        batch_size (int): _description_
        mbps (float): _description_
        lora (bool): _description_
        profile_dir (str, optional): used to save profile data during simulation. Defaults to 'log/sim_profile'.

    Returns:
        Tuple[MemoryConstant, TimeConstant]: _description_
    """
    try:
        layer_num = _get_layer_num(model_name=model)
    except ValueError as e:
        print(e)
        return None, None
    max_split_point = layer_num // 2
    CMD = ['bash', 'usl/simulate/profile.sh', str(mbps), str(batch_size), str(model), '--lora' if lora else '', str(max_split_point), profile_dir]
    need_profile = False
    for sp in SPLIT_POINTS:
        for offload in OFFLOAD_VALUES:
            # Generate the file name dynamically based on the parameters
            file_name = f"sp_{sp}_b_{batch_size}_mb_1_s_512_mbps_{mbps}_pipedream_wc{'_lora' if lora else '' }{'_oa_os' if offload else ''}.json"
            file_path = os.path.join(profile_dir, model, file_name)
            if not os.path.exists(file_path):
                need_profile = True
                break
        if need_profile:
            break
    if need_profile:
        res = subprocess.run(CMD)

    if not need_profile or (need_profile and res.returncode == 0):
        # if True:
        prof_res = {}
        print('Memory profile finished.')
        # Iterate over split_points and offload_values dynamically
        for sp in SPLIT_POINTS:
            prof_res[sp] = {}
            for offload in OFFLOAD_VALUES:
                # Generate the file name dynamically based on the parameters
                file_name = f"sp_{sp}_b_{batch_size}_mb_1_s_512_mbps_{mbps}_pipedream_wc{'_lora' if lora else '' }{'_oa_os' if offload else ''}.json"
                file_path = os.path.join(profile_dir, model, file_name)

                # Read and process the file
                try:
                    with open(file_path, 'r') as f:
                        data: dict = json.load(f)
                    # Store the parsed data
                    prof_res[sp][offload] = data
                except FileNotFoundError:
                    print(f"File {file_path} not found. Skipping.")
                    continue

        # MemoryVariable calculations
        mem_var = MemoryConstant(batch_size=batch_size)
        mem_var.no_off_mem_decrement_per_sp_server = round(
            prof_res[2][False]['server_max_mem_alloc_mb'] - prof_res[1][False]['server_max_mem_alloc_mb'], 2
        )
        mem_var.offload_mem_decrement_per_sp_server = round(
            prof_res[2][True]['server_max_mem_alloc_mb'] - prof_res[1][True]['server_max_mem_alloc_mb'], 2
        )
        mem_var.no_off_mem_increment_per_sp_client = round(
            prof_res[2][False]['client_max_mem_alloc_mb'] - prof_res[1][False]['client_max_mem_alloc_mb'], 2
        )
        mem_var.offload_mem_increment_per_sp_client = round(
            prof_res[2][True]['client_max_mem_alloc_mb'] - prof_res[1][True]['client_max_mem_alloc_mb'], 2
        )
        mem_var.base_max_mem_alloc_no_off_client = prof_res[1][False]['client_max_mem_alloc_mb']
        mem_var.base_max_mem_alloc_off_client = prof_res[1][True]['client_max_mem_alloc_mb']
        mem_var.base_max_mem_alloc_no_off_server = (
            prof_res[1][False]['server_max_mem_alloc_mb'] + (max_split_point - 2) * mem_var.no_off_mem_decrement_per_sp_server
        )
        mem_var.base_max_mem_alloc_off_server = (
            prof_res[1][True]['server_max_mem_alloc_mb'] + (max_split_point - 2) * mem_var.offload_mem_decrement_per_sp_server
        )
        # print(mem_var)

        # TimeVariable calculations
        time_var = TimeConstant(rate_mbps=mbps)
        time_var.base_no_off_head_fwd_time = prof_res[1][False]['head_fwd_time_avg_ms']
        time_var.base_no_off_head_bwd_time = prof_res[1][False]['head_bwd_time_avg_ms']
        time_var.base_off_head_fwd_time = prof_res[1][True]['head_fwd_time_avg_ms']
        time_var.base_off_head_bwd_time = prof_res[1][True]['head_bwd_time_avg_ms']
        time_var.base_tail_fwd_time = prof_res[1][False]['tail_fwd_time_avg_ms']
        time_var.base_tail_bwd_time = prof_res[1][False]['tail_bwd_time_avg_ms']
        time_var.head_activation_send_time = prof_res[1][False]['head_fwd_send_time_avg_ms']
        time_var.server_activation_send_time = prof_res[1][False]['server_fwd_send_time_avg_ms']
        time_var.tail_gradient_send_time = prof_res[1][False]['tail_bwd_send_time_avg_ms']
        time_var.server_gradient_send_time = prof_res[1][False]['server_bwd_send_time_avg_ms']
        time_var.head_no_off_fwd_time_increment_per_sp = round(
            prof_res[2][False]['head_fwd_time_avg_ms'] - prof_res[1][False]['head_fwd_time_avg_ms'], 2
        )
        time_var.head_no_off_bwd_time_increment_per_sp = round(
            prof_res[2][False]['head_bwd_time_avg_ms'] - prof_res[1][False]['head_bwd_time_avg_ms'], 2
        )
        time_var.head_off_fwd_time_increment_per_sp = round(prof_res[2][True]['head_fwd_time_avg_ms'] - prof_res[1][True]['head_fwd_time_avg_ms'], 2)
        time_var.head_off_bwd_time_increment_per_sp = round(prof_res[2][True]['head_bwd_time_avg_ms'] - prof_res[1][True]['head_bwd_time_avg_ms'], 2)
        time_var.server_no_off_fwd_time_increment_per_sp = round(
            prof_res[2][False]['server_fwd_time_avg_ms'] - prof_res[1][False]['server_fwd_time_avg_ms'], 2
        )
        time_var.server_no_off_bwd_time_increment_per_sp = round(
            prof_res[2][False]['server_bwd_time_avg_ms'] - prof_res[1][False]['server_bwd_time_avg_ms'], 2
        )
        time_var.server_off_fwd_time_increment_per_sp = round(
            prof_res[2][True]['server_fwd_time_avg_ms'] - prof_res[1][True]['server_fwd_time_avg_ms'], 2
        )
        time_var.server_off_bwd_time_increment_per_sp = round(
            prof_res[2][True]['server_bwd_time_avg_ms'] - prof_res[1][True]['server_bwd_time_avg_ms'], 2
        )
        time_var.tail_fwd_time_increment_per_sp = round(prof_res[2][False]['tail_fwd_time_avg_ms'] - prof_res[1][False]['tail_fwd_time_avg_ms'], 2)
        time_var.tail_bwd_time_increment_per_sp = round(prof_res[2][False]['tail_bwd_time_avg_ms'] - prof_res[1][False]['tail_bwd_time_avg_ms'], 2)

        # offload time
        head_m_off_time = prof_res[2][True]['head_m_offload_time_ms'] - prof_res[1][True]['head_m_offload_time_ms']
        head_os_off_time = prof_res[2][True]['head_os_offload_time_ms'] - prof_res[1][True]['head_os_offload_time_ms']
        tail_m_off_time = prof_res[2][True]['tail_m_offload_time_ms'] - prof_res[1][True]['tail_m_offload_time_ms']
        tail_os_off_time = prof_res[2][True]['tail_os_offload_time_ms'] - prof_res[1][True]['tail_os_offload_time_ms']
        time_var.base_head_offload_time = prof_res[1][True]['head_m_offload_time_ms'] + prof_res[1][True]['head_os_offload_time_ms']
        time_var.base_tail_offload_time = prof_res[1][True]['tail_m_offload_time_ms'] + prof_res[1][True]['tail_os_offload_time_ms']
        time_var.head_offload_time_increment_per_sp = head_m_off_time + head_os_off_time
        time_var.tail_offload_time_increment_per_sp = tail_m_off_time + tail_os_off_time
        time_var.base_no_off_server_fwd_time = (
            prof_res[1][False]['server_fwd_time_avg_ms'] + (max_split_point - 2) * time_var.server_no_off_fwd_time_increment_per_sp
        )
        time_var.base_no_off_server_bwd_time = (
            prof_res[1][False]['server_bwd_time_avg_ms'] + (max_split_point - 2) * time_var.server_no_off_bwd_time_increment_per_sp
        )
        time_var.base_off_server_fwd_time = (
            prof_res[1][True]['server_fwd_time_avg_ms'] + (max_split_point - 2) * time_var.server_off_fwd_time_increment_per_sp
        )
        time_var.base_off_server_bwd_time = (
            prof_res[1][True]['server_bwd_time_avg_ms'] + (max_split_point - 2) * time_var.server_off_bwd_time_increment_per_sp
        )
        # print(time_var)
        return mem_var, time_var

    else:
        print('Memory profile failed.')
        return None, None
