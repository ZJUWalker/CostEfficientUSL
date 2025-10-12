from dataclasses import dataclass, field
import json
import os
from typing import List, Any, Dict


@dataclass
class MainVariable:
    """
    MainVariable contains the variables needed to simulate the system.
    """

    split_point: int = 4
    offload: bool = True
    lora: bool = False


@dataclass
class TimeConstant:
    """
    TimeConstant contains the time cost of different components in the system.
    """

    rate_mbps: float = 1000  # mbps limit
    base_no_off_head_fwd_time: float = 8.99  # unit:ms
    base_off_head_fwd_time: float = 10.0  # unit:ms
    base_no_off_head_bwd_time: float = 17.55
    base_off_head_bwd_time: float = 19.0  # unit:ms
    head_no_off_fwd_time_increment_per_sp: float = 0.0  # unit:ms
    head_no_off_bwd_time_increment_per_sp: float = 0.0  # unit:ms
    head_off_fwd_time_increment_per_sp: float = 0.0  # unit:ms
    head_off_bwd_time_increment_per_sp: float = 0.0  # unit:ms
    base_no_off_server_fwd_time: float = 43.14
    base_no_off_server_bwd_time: float = 87.83
    base_off_server_fwd_time: float = 43.14
    base_off_server_bwd_time: float = 87.83
    server_off_fwd_time_increment_per_sp: float = 0.0  # unit:ms
    server_off_bwd_time_increment_per_sp: float = 0.0  # unit:ms
    server_no_off_fwd_time_increment_per_sp: float = 0.0  # unit:ms
    server_no_off_bwd_time_increment_per_sp: float = 0.0  # unit:ms
    head_activation_send_time: float = 44.11  # unit:ms
    tail_gradient_send_time: float = 33.41
    # most of the time,server_activation_recv_time approximates head_gradient_recv_time and server gradient send time
    server_activation_send_time: float = 33.93
    server_gradient_send_time: float = 33.65  # unit:ms
    base_tail_fwd_time: float = 11.43
    base_tail_bwd_time: float = 28.63  # unit:ms
    tail_fwd_time_increment_per_sp: float = 0.0  # unit:ms
    tail_bwd_time_increment_per_sp: float = 0.0  # unit:ms
    # offload and reload time
    base_head_offload_time: float = 10.0  # unit:ms
    head_offload_time_increment_per_sp: float = 0.0  # unit:ms
    base_tail_offload_time: float = 10.0  # unit:ms
    tail_offload_time_increment_per_sp: float = 0.0  # unit:ms


@dataclass
class MemoryConstant:
    """
    MemoryConstant contains the memory usage of different components in the system.
    """

    batch_size: int = 8
    micro_batch_size: int = 1
    max_seq_len: int = 512
    # do four profile, (sp=1, no off),(sp=2 ,no off),(sp=1 ,off),(sp=2 ,off)
    base_max_mem_alloc_no_off_client: float = 9899.5542  # unit:MB # sp=1，不做卸载的时候的最大显存分配
    base_max_mem_alloc_off_client: float = 6804.4971  # unit:MB # sp=1，做卸载的时候的最大显存分配
    base_max_mem_alloc_no_off_server: float = 23328.3828  # unit:MB # sp=1，不做卸载的时候的最大显存分配
    base_max_mem_alloc_off_server: float = 23328.3828  # unit:MB # sp=1，不做卸载的时候的最大显存分配
    no_off_mem_increment_per_sp_client: float = 2560.70  # 不做卸载，每多一个sp，显存的增加量
    no_off_mem_decrement_per_sp_server: float = 3313.054  # 不做卸载，每多一个sp，服务端显存的减少量
    offload_mem_increment_per_sp_client: float = 1728.1601  # unit:MB，如果做卸载，每加一个sp，最大显存分配减少的量
    offload_mem_decrement_per_sp_server: float = 1728.1601  # unit:MB，如果做卸载，每加一个sp，服务端最大显存分配减少的量


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

    def save_to_json(self, file_path='log/simulate/simulate_result.json'):
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
