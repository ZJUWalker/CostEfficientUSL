import subprocess
import json
import os
import argparse
from typing import Tuple
from usl.simulate import MemoryConstant, TimeConstant

SPLIT_POINTS = [1, 2]
OFFLOAD_VALUES = [False, True]


def run_profile(model: str, batch_size: int, mbps: float, lora: bool, profile_dir: str = 'log/profile') -> Tuple[MemoryConstant, TimeConstant]:
    CMD = ['bash', 'usl/simulate/profile.sh', str(mbps), str(batch_size), str(model), '--lora' if lora else '']
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
        mem_var.base_max_mem_alloc_no_off_client = prof_res[1][False]['client_max_mem_alloc_mb']
        mem_var.base_max_mem_alloc_off_client = prof_res[1][True]['client_max_mem_alloc_mb']
        mem_var.base_max_mem_alloc_no_off_server = prof_res[1][False]['server_max_mem_alloc_mb']
        mem_var.base_max_mem_alloc_off_server = prof_res[1][True]['server_max_mem_alloc_mb']
        mem_var.base_mem_decrement_per_sp_server = round(
            prof_res[1][False]['server_max_mem_alloc_mb'] - prof_res[2][False]['server_max_mem_alloc_mb'], 2
        )
        mem_var.no_off_mem_increment_per_sp_client = round(
            prof_res[2][False]['client_max_mem_alloc_mb'] - prof_res[1][False]['client_max_mem_alloc_mb'], 2
        )
        mem_var.offload_mem_increment_per_sp_client = round(
            prof_res[2][True]['client_max_mem_alloc_mb'] - prof_res[1][True]['client_max_mem_alloc_mb'], 2
        )
        # print(mem_var)

        # TimeVariable calculations
        time_var = TimeConstant(rate_mbps=mbps)
        time_var.base_head_fwd_time = prof_res[1][False]['head_fwd_time_avg_ms']
        time_var.base_head_bwd_time = prof_res[1][False]['head_bwd_time_avg_ms']
        time_var.base_server_fwd_time = prof_res[1][False]['server_fwd_time_avg_ms']
        time_var.base_server_bwd_time = prof_res[1][False]['server_bwd_time_avg_ms']
        time_var.base_tail_fwd_time = prof_res[1][False]['tail_fwd_time_avg_ms']
        time_var.base_tail_bwd_time = prof_res[1][False]['tail_bwd_time_avg_ms']
        time_var.head_activation_send_time = prof_res[1][False]['head_fwd_send_time_avg_ms']
        time_var.server_activation_send_time = prof_res[1][False]['server_fwd_send_time_avg_ms']
        time_var.tail_gradient_send_time = prof_res[1][False]['tail_bwd_send_time_avg_ms']
        time_var.server_gradient_send_time = prof_res[1][False]['server_bwd_send_time_avg_ms']
        time_var.head_fwd_time_increment_per_sp = round(prof_res[2][False]['head_fwd_time_avg_ms'] - prof_res[1][False]['head_fwd_time_avg_ms'], 2)
        time_var.head_bwd_time_increment_per_sp = round(prof_res[2][False]['head_bwd_time_avg_ms'] - prof_res[1][False]['head_bwd_time_avg_ms'], 2)
        time_var.server_fwd_time_decrement_per_sp = round(
            prof_res[1][False]['server_fwd_time_avg_ms'] - prof_res[2][False]['server_fwd_time_avg_ms'], 2
        )
        time_var.server_bwd_time_decrement_per_sp = round(
            prof_res[1][False]['server_bwd_time_avg_ms'] - prof_res[2][False]['server_bwd_time_avg_ms'], 2
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
        # print(time_var)
        return mem_var, time_var

    else:
        print('Memory profile failed.')
        return None, None
