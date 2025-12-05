from dataclasses import dataclass, field
import json
import os
from typing import List, Any, Dict


@dataclass
class MainVariable:
    """
    MainVariable contains the variables needed to simulate the system.
    """

    total_sample_count: int = 10000  # total sample count need to be trained per epoch
    batch_size: int = 8  # batch size per batch
    split_point: int = 4
    gpu_max_capacity: int = 48  # unit:GB
    gpu_rent_cost_per_hour: float = 1.0  # unit:USD/hour
    client_offload_mb_num: int = 0
    server_offload_mb_num: int = 0
    client_offload_model_state_sp_num: int = 0
    lora: bool = True  # our experiment only use lora ft for now,but also support non-lora ft


@dataclass
class TimeConstant:
    """
    TimeConstant contains the time cost of different components in the system.All of time unit is in millisecond.
    """

    rate_mbps: float = 1000  # mbps limit
    base_head_fwd_time_per_mb: float = 0.0
    base_head_bwd_time_per_mb: float = 0.0
    base_server_fwd_time_per_mb: float = 0.0
    base_server_bwd_time_per_mb: float = 0.0
    base_tail_fwd_time_per_mb: float = 0.0
    base_tail_bwd_time_per_mb: float = 0.0
    head_fwd_time_increment_per_sp: float = 0.0
    head_bwd_time_increment_per_sp: float = 0.0
    server_fwd_time_increment_per_sp: float = 0.0
    server_bwd_time_increment_per_sp: float = 0.0
    tail_fwd_time_increment_per_sp: float = 0.0
    tail_bwd_time_increment_per_sp: float = 0.0
    # activation offload and reload time
    base_head_activation_offload_time_per_mb: float = 0.0
    base_head_activation_reload_time_per_mb: float = 0.0
    base_server_activation_offload_time_per_mb: float = 0.0
    base_server_activation_reload_time_per_mb: float = 0.0
    head_activation_offload_time_increment_per_sp: float = 0.0
    server_activation_offload_time_increment_per_sp: float = 0.0
    head_activation_reload_time_increment_per_sp: float = 0.0
    server_activation_reload_time_increment_per_sp: float = 0.0
    # model offload and reload time
    base_head_model_state_offload_time: float = 0.0
    base_tail_model_state_offload_time: float = 0.0
    head_model_offload_time_increment_per_sp: float = 0.0
    tail_model_offload_time_increment_per_sp: float = 0.0
    # idle time between two compute and communication
    delay_time_avg_ms: float = 0.0
    # gradient offload and reload time
    head_activation_send_time: float = 0
    tail_gradient_send_time: float = 0
    server_activation_send_time: float = 0
    server_gradient_send_time: float = 0


@dataclass
class MemoryConstant:
    """
    MemoryConstant contains the memory usage of different components in the system.
    """

    max_split_point: int = 18  # max split point of the model,(qwen3-4b->18, qwen3-8b->18, qwen3-14b->20, qwen3-32b->36),auto set
    micro_batch_size: int = 1
    max_seq_len: int = 512
    baseline_split_point: int = 1
    baseline_minibatch_num: int = 4
    base_client_mem_alloc: float = 1024.0  # unit:MB
    base_server_mem_alloc: float = 1024.0  # unit:MB
    mem_increment_per_sp_client: float = 0.0
    mem_increment_per_mb_client_if_oa: float = 0.0
    mem_increment_per_sp_server: float = 0.0
    mem_increment_per_mb_server_if_oa: float = 0.0
    mem_increment_per_sp_mb_client: float = 0.0
    mem_increment_per_sp_mb_server: float = 0.0
    base_model_state_mem_alloc_client: float = 0
    base_model_state_mem_alloc_except_blocks: float = 0
    model_mem_increment_per_sp_client: float = 0  # unit:MB，如果做卸载，每加一个sp，最大显存分配减少的量
    nccl_buffer_per_rank_per_mb: float = 10.0  # unit:MB for each rank (about 30-300MB)
    nccl_base_buffer_size: float = 80.0  # unit:MB for each rank,base nccl buffer size


@dataclass
class Objective:
    """
    Objective contains the objective function values of different components in the system,used to do the optimization.
    """

    client_peak_mem_alloc: float = 0.0  # unit:MB
    server_peak_mem_alloc: float = 0.0
    batch_train_time: float = 0.0  # unit:ms
    server_cost_per_epoch: float = 0.0  # unit:USD/epoch
    min_gpu_num_required: int = 0
    layers_num_per_gpu: list = field(default_factory=lambda: [])  # for debug
    mem_alloc_per_gpu: list = field(default_factory=lambda: [])  # for debug
    mem_alloc_per_layer: float = 0.0  # unit:MB for each layer
    epoch_train_time: float = 0.0  # unit:hour
    client_idle_rate: float = 0.0  # unit:%
    server_idle_rate: float = 0.0
    client_send_rate: float = 0.0
    server_send_rate: float = 0.0
    pass


@dataclass
class SimulateResult:
    """
    SimulateResult contains the result of the simulation.
    """

    model_name: str = "default_model"
    main_variable: MainVariable = field(default_factory=lambda: MainVariable())
    time_const: TimeConstant = field(default_factory=lambda: TimeConstant())
    memory_const: MemoryConstant = field(default_factory=lambda: MemoryConstant())
    objective: Objective = field(default_factory=lambda: Objective())

    def to_dict(self):
        return {
            "model": self.model_name,
            **self.main_variable.__dict__,
            **self.objective.__dict__,
            **self.time_const.__dict__,
            **self.memory_const.__dict__,
        }

    def to_simple_dict(self):
        return {
            "model": self.model_name,
            **self.main_variable.__dict__,
            **self.objective.__dict__,
        }

    def save_to_json(self, file_path='log/simulate/simulate_result.json'):
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
