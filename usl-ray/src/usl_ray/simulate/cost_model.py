"""Profile-derived constants used by the analytical simulator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MainVariable:
    """Search variables and deployment constraints."""

    total_sample_count: int = 10_000
    batch_size: int = 8
    split_point: int = 4
    gpu_max_capacity: int = 48
    gpu_rent_cost_per_hour: float = 1.0
    client_offload_mb_num: int = 0
    server_offload_mb_num: int = 0
    client_offload_model_state_sp_num: int = 0
    lora: bool = True


@dataclass(frozen=True)
class TimeCost:
    """Operation durations in milliseconds."""

    rate_mbps: float = 1000.0
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
    base_head_activation_offload_time_per_mb: float = 0.0
    base_head_activation_reload_time_per_mb: float = 0.0
    base_server_activation_offload_time_per_mb: float = 0.0
    base_server_activation_reload_time_per_mb: float = 0.0
    head_activation_offload_time_increment_per_sp: float = 0.0
    server_activation_offload_time_increment_per_sp: float = 0.0
    head_activation_reload_time_increment_per_sp: float = 0.0
    server_activation_reload_time_increment_per_sp: float = 0.0
    base_head_model_state_offload_time: float = 0.0
    base_tail_model_state_offload_time: float = 0.0
    head_model_offload_time_increment_per_sp: float = 0.0
    tail_model_offload_time_increment_per_sp: float = 0.0
    delay_time_avg_ms: float = 0.0
    head_activation_send_time: float = 0.0
    tail_gradient_send_time: float = 0.0
    server_activation_send_time: float = 0.0
    server_gradient_send_time: float = 0.0


@dataclass(frozen=True)
class MemoryCost:
    """Memory constants in MiB."""

    max_split_point: int = 18
    micro_batch_size: int = 1
    max_seq_len: int = 512
    baseline_split_point: int = 1
    baseline_minibatch_num: int = 4
    base_client_mem_alloc: float = 1024.0
    base_server_mem_alloc: float = 1024.0
    mem_increment_per_sp_client: float = 0.0
    mem_increment_per_mb_client_if_oa: float = 0.0
    mem_increment_per_sp_server: float = 0.0
    mem_increment_per_mb_server_if_oa: float = 0.0
    mem_increment_per_sp_mb_client: float = 0.0
    mem_increment_per_sp_mb_server: float = 0.0
    base_model_state_mem_alloc_client: float = 0.0
    base_model_state_mem_alloc_except_blocks: float = 0.0
    model_mem_increment_per_sp_client: float = 0.0
    nccl_buffer_per_rank_per_mb: float = 10.0
    nccl_base_buffer_size: float = 80.0


@dataclass(frozen=True)
class CostModel:
    """Persisted analytical model."""

    time: TimeCost
    memory: MemoryCost

    def save(self, path: str | Path) -> Path:
        """Write a reproducible cost model."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> CostModel:
        """Load one cost model."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(TimeCost(**payload["time"]), MemoryCost(**payload["memory"]))


def _operation(profile: dict[str, Any], name: str) -> float:
    return float(profile["metrics"]["operation_mean_ms"].get(name, 0.0))


def fit_from_profiles(
    profiles: dict[tuple[int, int, int, int, int], dict[str, Any]],
    *,
    base_split_point: int,
    base_batch_size: int,
    max_split_point: int,
    micro_batch_size: int,
    max_seq_len: int,
    lora: bool,
) -> CostModel:
    """Fit legacy-compatible differences from seven calibration profiles."""
    required = {
        (base_split_point, base_batch_size, 0, 0, 0),
        (base_split_point, base_batch_size, base_batch_size, 0, base_batch_size),
        (base_split_point + 1, base_batch_size, 0, 0, 0),
        (base_split_point + 1, base_batch_size, base_batch_size, 0, base_batch_size),
        (base_split_point, base_batch_size, 0, base_split_point, 0),
        (base_split_point + 1, base_batch_size, 0, base_split_point + 1, 0),
        (base_split_point, base_batch_size + 1, base_batch_size + 1, 0, base_batch_size + 1),
    }
    missing = sorted(required - profiles.keys())
    if missing:
        raise ValueError(f"Missing calibration combinations: {missing}")
    base_key = (base_split_point, base_batch_size, 0, 0, 0)
    next_key = (base_split_point + 1, base_batch_size, 0, 0, 0)
    base = profiles[base_key]
    next_split = profiles[next_key]
    activation_key = (base_split_point, base_batch_size, base_batch_size, 0, base_batch_size)
    next_activation_key = (
        base_split_point + 1,
        base_batch_size,
        base_batch_size,
        0,
        base_batch_size,
    )
    model_key = (base_split_point, base_batch_size, 0, base_split_point, 0)
    next_model_key = (
        base_split_point + 1,
        base_batch_size,
        0,
        base_split_point + 1,
        0,
    )
    larger_batch_key = (
        base_split_point,
        base_batch_size + 1,
        base_batch_size + 1,
        0,
        base_batch_size + 1,
    )
    activation = profiles[activation_key]
    next_activation = profiles[next_activation_key]
    model_offload = profiles[model_key]
    next_model_offload = profiles[next_model_key]
    larger_batch = profiles[larger_batch_key]
    base_client = float(base["client_max_memory_mib"])
    base_server = max(float(item) for item in base["server_max_memory_mib"])
    next_client = float(next_split["client_max_memory_mib"])
    next_server = max(float(item) for item in next_split["server_max_memory_mib"])
    activation_client = float(activation["client_max_memory_mib"])
    activation_server = max(float(item) for item in activation["server_max_memory_mib"])
    next_activation_client = float(next_activation["client_max_memory_mib"])
    next_activation_server = max(float(item) for item in next_activation["server_max_memory_mib"])
    model_client = float(model_offload["client_max_memory_mib"])
    next_model_client = float(next_model_offload["client_max_memory_mib"])
    larger_client = float(larger_batch["client_max_memory_mib"])
    larger_server = max(float(item) for item in larger_batch["server_max_memory_mib"])
    head_activation_offload = max(0.0, _operation(activation, "head_forward") - _operation(base, "head_forward"))
    next_head_activation_offload = max(
        0.0,
        _operation(next_activation, "head_forward") - _operation(next_split, "head_forward"),
    )
    head_activation_reload = max(
        0.0,
        _operation(activation, "head_backward") - _operation(base, "head_backward"),
    )
    next_head_activation_reload = max(
        0.0,
        _operation(next_activation, "head_backward") - _operation(next_split, "head_backward"),
    )
    server_activation_offload = max(
        0.0,
        _operation(activation, "server_forward") - _operation(base, "server_forward"),
    )
    next_server_activation_offload = max(
        0.0,
        _operation(next_activation, "server_forward") - _operation(next_split, "server_forward"),
    )
    server_activation_reload = max(
        0.0,
        _operation(activation, "server_backward") - _operation(base, "server_backward"),
    )
    next_server_activation_reload = max(
        0.0,
        _operation(next_activation, "server_backward") - _operation(next_split, "server_backward"),
    )
    timing = TimeCost(
        rate_mbps=float(base["config"]["network"]["mbps"]),
        base_head_fwd_time_per_mb=_operation(base, "head_forward"),
        base_head_bwd_time_per_mb=_operation(base, "head_backward"),
        base_server_fwd_time_per_mb=_operation(base, "server_forward"),
        base_server_bwd_time_per_mb=_operation(base, "server_backward"),
        base_tail_fwd_time_per_mb=_operation(base, "tail_forward"),
        base_tail_bwd_time_per_mb=_operation(base, "tail_backward"),
        head_fwd_time_increment_per_sp=_operation(next_split, "head_forward") - _operation(base, "head_forward"),
        head_bwd_time_increment_per_sp=_operation(next_split, "head_backward") - _operation(base, "head_backward"),
        server_fwd_time_increment_per_sp=_operation(next_split, "server_forward") - _operation(base, "server_forward"),
        server_bwd_time_increment_per_sp=_operation(next_split, "server_backward")
        - _operation(base, "server_backward"),
        tail_fwd_time_increment_per_sp=_operation(next_split, "tail_forward") - _operation(base, "tail_forward"),
        tail_bwd_time_increment_per_sp=_operation(next_split, "tail_backward") - _operation(base, "tail_backward"),
        base_head_activation_offload_time_per_mb=head_activation_offload,
        base_head_activation_reload_time_per_mb=head_activation_reload,
        base_server_activation_offload_time_per_mb=server_activation_offload,
        base_server_activation_reload_time_per_mb=server_activation_reload,
        head_activation_offload_time_increment_per_sp=next_head_activation_offload - head_activation_offload,
        head_activation_reload_time_increment_per_sp=next_head_activation_reload - head_activation_reload,
        server_activation_offload_time_increment_per_sp=next_server_activation_offload - server_activation_offload,
        server_activation_reload_time_increment_per_sp=next_server_activation_reload - server_activation_reload,
        base_head_model_state_offload_time=_operation(model_offload, "head_model_offload"),
        base_tail_model_state_offload_time=_operation(model_offload, "tail_model_offload"),
        head_model_offload_time_increment_per_sp=(
            _operation(next_model_offload, "head_model_offload") - _operation(model_offload, "head_model_offload")
        ),
        tail_model_offload_time_increment_per_sp=(
            _operation(next_model_offload, "tail_model_offload") - _operation(model_offload, "tail_model_offload")
        ),
        delay_time_avg_ms=float(base["config"]["network"]["delay_ms"]),
        head_activation_send_time=_operation(base, "act_up"),
        tail_gradient_send_time=_operation(base, "grad_up"),
        server_activation_send_time=_operation(base, "act_down"),
        server_gradient_send_time=_operation(base, "grad_down"),
    )
    model_state_memory = base_client - model_client
    model_state_increment = (next_client - next_model_client) - model_state_memory
    memory = MemoryCost(
        max_split_point=max_split_point,
        micro_batch_size=micro_batch_size,
        max_seq_len=max_seq_len,
        baseline_split_point=base_split_point,
        baseline_minibatch_num=base_batch_size,
        base_client_mem_alloc=base_client,
        base_server_mem_alloc=base_server + (max_split_point - 2 * base_split_point) * (next_server - base_server),
        mem_increment_per_sp_client=next_client - base_client,
        mem_increment_per_sp_server=next_server - base_server,
        mem_increment_per_mb_client_if_oa=larger_client - activation_client,
        mem_increment_per_mb_server_if_oa=larger_server - activation_server,
        mem_increment_per_sp_mb_client=(
            (next_client - next_activation_client) / max(1, base_batch_size - 1) / (base_split_point + 1)
        ),
        mem_increment_per_sp_mb_server=(
            (next_server - next_activation_server)
            / max(1, base_batch_size - (2 if lora else 3))
            / (base_split_point + 1)
        ),
        base_model_state_mem_alloc_client=model_state_memory,
        base_model_state_mem_alloc_except_blocks=model_state_memory - base_split_point * model_state_increment,
        model_mem_increment_per_sp_client=model_state_increment,
        nccl_buffer_per_rank_per_mb=2 * base["config"]["model"]["hidden_size"] * 4 * max_seq_len / 1024**2,
    )
    return CostModel(timing, memory)
