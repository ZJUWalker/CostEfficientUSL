import subprocess
import json
import os
import argparse
from typing import Tuple
from usl.simulate import MemoryConstant, TimeConstant
from transformers import AutoConfig

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
    batch_sizes = [base_bs, base_bs + 1]
    split_points = [base_sp, base_sp + 1]
    CMD = [
        'bash',
        'usl/simulate/profile.sh',
        str(mbps),
        str(model),
        '--lora' if lora else '',
        str(max_split_point),
        ' '.join([str(i) for i in batch_sizes]),
        ' '.join([str(i) for i in split_points]),
        profile_dir,
    ]
    need_profile = False
    for bs in batch_sizes:
        for sp in split_points:
            for offload in OFFLOAD_VALUES:
                # Generate the file name dynamically based on the parameters
                file_name = f"sp_{sp}_b_{bs}_mb_1_s_512_mbps_{mbps}_pipedream_wc{'_lora' if lora else '' }{f'_oa_{bs}_os_1.0' if offload else ''}.json"
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
        for bs in batch_sizes:
            prof_res[bs] = {}
            for sp in split_points:
                prof_res[bs][sp] = {}
                for offload in OFFLOAD_VALUES:
                    # Generate the file name dynamically based on the parameters
                    file_name = f"sp_{sp}_b_{bs}_mb_1_s_512_mbps_{mbps}_pipedream_wc{'_lora' if lora else '' }{f'_oa_{bs}_os_1.0' if offload else ''}.json"
                    file_path = os.path.join(profile_dir, model, file_name)

                    # Read and process the file
                    try:
                        with open(file_path, 'r') as f:
                            data: dict = json.load(f)
                        # Store the parsed data
                        prof_res[bs][sp][offload] = data
                    except FileNotFoundError:
                        print(f"File {file_path} not found. Skipping.")
                        continue

        # MemoryVariable calculations
        mem_var = MemoryConstant(max_split_point=max_split_point, baseline_split_point=base_sp, baseline_minibatch_num=base_bs)
        mem_var.no_off_mem_decrement_per_sp_server = round(
            prof_res[base_bs][base_sp + 1][False]['server_max_mem_alloc_mb'] - prof_res[base_bs][base_sp][False]['server_max_mem_alloc_mb'], 2
        )  # checked
        mem_var.offload_mem_decrement_per_sp_server = round(
            prof_res[base_bs][base_sp + 1][True]['server_max_mem_alloc_mb'] - prof_res[base_bs][base_sp][True]['server_max_mem_alloc_mb'], 2
        )  # a bit of loss
        mem_var.no_off_mem_increment_per_sp_client = round(
            prof_res[base_bs][base_sp + 1][False]['client_max_mem_alloc_mb'] - prof_res[base_bs][base_sp][False]['client_max_mem_alloc_mb'], 2
        )
        mem_var.offload_mem_increment_per_sp_client = round(
            prof_res[base_bs][base_sp + 1][True]['client_max_mem_alloc_mb'] - prof_res[base_bs][base_sp][True]['client_max_mem_alloc_mb'], 2
        )
        mem_var.base_max_mem_alloc_no_off_client = prof_res[base_bs][base_sp][False]['client_max_mem_alloc_mb']
        mem_var.base_max_mem_alloc_off_client = prof_res[base_bs][base_sp][True]['client_max_mem_alloc_mb']
        mem_var.base_max_mem_alloc_no_off_server = (
            prof_res[base_bs][base_sp][False]['server_max_mem_alloc_mb']
            + (max_split_point - base_sp * 2) * mem_var.no_off_mem_decrement_per_sp_server
        )  # checked
        mem_var.base_max_mem_alloc_off_server = (
            prof_res[base_bs][base_sp][True]['server_max_mem_alloc_mb']
            + (max_split_point - base_sp * 2) * mem_var.offload_mem_decrement_per_sp_server
        )
        mem_var.offload_mem_increment_per_mb_client = (
            prof_res[base_bs + 1][base_sp][True]['client_max_mem_alloc_mb'] - prof_res[base_bs][base_sp][True]['client_max_mem_alloc_mb']
        )
        mem_var.no_off_mem_increment_per_mb_client = (
            (prof_res[base_bs + 1][base_sp][False]['client_max_mem_alloc_mb'] - prof_res[base_bs][base_sp][False]['client_max_mem_alloc_mb'])
            - mem_var.offload_mem_increment_per_mb_client
        ) / mem_var.baseline_split_point

        mem_var.no_off_mem_increment_per_mb_server = (
            (prof_res[base_bs + 1][base_sp][False]['server_max_mem_alloc_mb'] - prof_res[base_bs][base_sp][False]['server_max_mem_alloc_mb'])
            - mem_var.offload_mem_increment_per_mb_client
        ) / mem_var.baseline_split_point  # checked
        mem_var.offload_mem_increment_per_mb_server = 0  # checked
        print(mem_var)

        # TimeVariable calculations
        time_var = TimeConstant(rate_mbps=mbps)
        time_var.base_no_off_head_fwd_time = prof_res[base_bs][base_sp][False]['head_fwd_time_avg_ms']
        time_var.base_no_off_head_bwd_time = prof_res[base_bs][base_sp][False]['head_bwd_time_avg_ms']
        time_var.base_off_head_fwd_time = prof_res[base_bs][base_sp][True]['head_fwd_time_avg_ms']
        time_var.base_off_head_bwd_time = prof_res[base_bs][base_sp][True]['head_bwd_time_avg_ms']
        time_var.base_tail_fwd_time = prof_res[base_bs][base_sp][False]['tail_fwd_time_avg_ms']
        time_var.base_tail_bwd_time = prof_res[base_bs][base_sp][False]['tail_bwd_time_avg_ms']
        time_var.head_activation_send_time = prof_res[base_bs][base_sp][False]['head_fwd_send_time_avg_ms']
        time_var.server_activation_send_time = prof_res[base_bs][base_sp][False]['server_fwd_send_time_avg_ms']
        time_var.tail_gradient_send_time = prof_res[base_bs][base_sp][False]['tail_bwd_send_time_avg_ms']
        time_var.server_gradient_send_time = prof_res[base_bs][base_sp][False]['server_bwd_send_time_avg_ms']
        time_var.head_no_off_fwd_time_increment_per_sp = round(
            prof_res[base_bs][base_sp + 1][False]['head_fwd_time_avg_ms'] - prof_res[base_bs][base_sp][False]['head_fwd_time_avg_ms'], 2
        )
        time_var.head_no_off_bwd_time_increment_per_sp = round(
            prof_res[base_bs][base_sp + 1][False]['head_bwd_time_avg_ms'] - prof_res[base_bs][base_sp][False]['head_bwd_time_avg_ms'], 2
        )
        time_var.head_off_fwd_time_increment_per_sp = round(
            prof_res[base_bs][base_sp + 1][True]['head_fwd_time_avg_ms'] - prof_res[base_bs][base_sp][True]['head_fwd_time_avg_ms'], 2
        )
        time_var.head_off_bwd_time_increment_per_sp = round(
            prof_res[base_bs][base_sp + 1][True]['head_bwd_time_avg_ms'] - prof_res[base_bs][base_sp][True]['head_bwd_time_avg_ms'], 2
        )
        time_var.server_no_off_fwd_time_increment_per_sp = round(
            prof_res[base_bs][base_sp + 1][False]['server_fwd_time_avg_ms'] - prof_res[base_bs][base_sp][False]['server_fwd_time_avg_ms'], 2
        )
        time_var.server_no_off_bwd_time_increment_per_sp = round(
            prof_res[base_bs][base_sp + 1][False]['server_bwd_time_avg_ms'] - prof_res[base_bs][base_sp][False]['server_bwd_time_avg_ms'], 2
        )
        time_var.server_off_fwd_time_increment_per_sp = round(
            prof_res[base_bs][base_sp + 1][True]['server_fwd_time_avg_ms'] - prof_res[base_bs][base_sp][True]['server_fwd_time_avg_ms'], 2
        )
        time_var.server_off_bwd_time_increment_per_sp = round(
            prof_res[base_bs][base_sp + 1][True]['server_bwd_time_avg_ms'] - prof_res[base_bs][base_sp][True]['server_bwd_time_avg_ms'], 2
        )
        time_var.tail_fwd_time_increment_per_sp = round(
            prof_res[base_bs][base_sp + 1][False]['tail_fwd_time_avg_ms'] - prof_res[base_bs][base_sp][False]['tail_fwd_time_avg_ms'], 2
        )
        time_var.tail_bwd_time_increment_per_sp = round(
            prof_res[base_bs][base_sp + 1][False]['tail_bwd_time_avg_ms'] - prof_res[base_bs][base_sp][False]['tail_bwd_time_avg_ms'], 2
        )

        # offload time
        head_m_off_time = prof_res[base_bs][base_sp + 1][True]['head_m_offload_time_ms'] - prof_res[base_bs][base_sp][True]['head_m_offload_time_ms']
        head_os_off_time = (
            prof_res[base_bs][base_sp + 1][True]['head_os_offload_time_ms'] - prof_res[base_bs][base_sp][True]['head_os_offload_time_ms']
        )
        tail_m_off_time = prof_res[base_bs][base_sp + 1][True]['tail_m_offload_time_ms'] - prof_res[base_bs][base_sp][True]['tail_m_offload_time_ms']
        tail_os_off_time = (
            prof_res[base_bs][base_sp + 1][True]['tail_os_offload_time_ms'] - prof_res[base_bs][base_sp][True]['tail_os_offload_time_ms']
        )
        time_var.base_head_offload_time = (
            prof_res[base_bs][base_sp][True]['head_m_offload_time_ms'] + prof_res[base_bs][base_sp][True]['head_os_offload_time_ms']
        )
        time_var.base_tail_offload_time = (
            prof_res[base_bs][base_sp][True]['tail_m_offload_time_ms'] + prof_res[base_bs][base_sp][True]['tail_os_offload_time_ms']
        )
        time_var.head_offload_time_increment_per_sp = head_m_off_time + head_os_off_time
        time_var.tail_offload_time_increment_per_sp = tail_m_off_time + tail_os_off_time
        time_var.base_no_off_server_fwd_time = (
            prof_res[base_bs][base_sp][False]['server_fwd_time_avg_ms']
            + (max_split_point - base_sp - 1) * time_var.server_no_off_fwd_time_increment_per_sp
        )
        time_var.base_no_off_server_bwd_time = (
            prof_res[base_bs][base_sp][False]['server_bwd_time_avg_ms']
            + (max_split_point - base_sp - 1) * time_var.server_no_off_bwd_time_increment_per_sp
        )
        time_var.base_off_server_fwd_time = (
            prof_res[base_bs][base_sp][True]['server_fwd_time_avg_ms']
            + (max_split_point - base_sp - 1) * time_var.server_off_fwd_time_increment_per_sp
        )
        time_var.base_off_server_bwd_time = (
            prof_res[base_bs][base_sp][True]['server_bwd_time_avg_ms']
            + (max_split_point - base_sp - 1) * time_var.server_off_bwd_time_increment_per_sp
        )
        time_var.delay_time_avg_ms = (
            prof_res[base_bs][base_sp][False]['delay_time_avg_ms'] + prof_res[base_bs][base_sp][True]['delay_time_avg_ms']
        ) / 2
        # print(time_var)
        return mem_var, time_var

    else:
        print('Memory profile failed.')
        return None, None
