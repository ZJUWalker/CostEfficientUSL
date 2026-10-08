"""Deterministic SplitMind memory and timeline simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

from usl_ray.profile.link import LinkModel
from usl_ray.profile.timeline import Span, StepMetrics, StepTimeline, summarize
from usl_ray.simulate.cost_model import MainVariable, MemoryCost, TimeCost


@dataclass(frozen=True)
class Objective:
    """Simulation objectives using legacy field names and units."""

    client_peak_mem_alloc: float = 0.0
    server_peak_mem_alloc: float = 0.0
    batch_train_time: float = 0.0
    server_cost_per_epoch: float = 0.0
    min_gpu_num_required: int = 0
    layers_num_per_gpu: tuple[int, ...] = ()
    mem_alloc_per_gpu: tuple[float, ...] = ()
    mem_alloc_per_layer: float = 0.0
    epoch_train_time: float = 0.0
    client_idle_rate: float = 0.0
    server_idle_rate: float = 0.0
    client_send_rate: float = 0.0
    server_send_rate: float = 0.0
    feasible: bool = True


@dataclass(frozen=True)
class SimulationResult:
    """Simulated timeline and derived objective."""

    variables: MainVariable
    objective: Objective
    timeline: StepTimeline = field(default_factory=lambda: StepTimeline(0))
    metrics: StepMetrics = field(default_factory=lambda: summarize([]))


def _simulate_memory(variables: MainVariable, memory: MemoryCost) -> Objective:
    split = variables.split_point
    batch = variables.batch_size
    base_split = memory.baseline_split_point
    base_batch = memory.baseline_minibatch_num
    max_split = memory.max_split_point
    client = (
        memory.base_client_mem_alloc
        + (split - base_split) * memory.mem_increment_per_sp_client
        + (batch - base_batch) * split * memory.mem_increment_per_sp_mb_client
        - split * max(0, variables.client_offload_mb_num - 1) * memory.mem_increment_per_sp_mb_client
    )
    if variables.client_offload_model_state_sp_num > 0:
        client -= (
            memory.base_model_state_mem_alloc_except_blocks
            + variables.client_offload_model_state_sp_num * memory.model_mem_increment_per_sp_client
        )
    server = (
        memory.base_server_mem_alloc
        - (split - base_split) * memory.mem_increment_per_sp_server
        + (batch - base_batch) * (max_split - split) * memory.mem_increment_per_sp_mb_server
        - (max_split - split)
        * max(0, variables.server_offload_mb_num - (2 if variables.lora else 3))
        * memory.mem_increment_per_sp_mb_server
    )
    server_layers = (max_split - split) * 2
    if server_layers <= 0:
        return Objective(client_peak_mem_alloc=client, server_peak_mem_alloc=server, feasible=False)
    for gpu_count in range(1, 31):
        with_nccl = server + (memory.nccl_base_buffer_size + memory.nccl_buffer_per_rank_per_mb * batch) * (
            gpu_count - 1
        )
        if with_nccl <= gpu_count * variables.gpu_max_capacity * 1024 * 0.95:
            per_layer = with_nccl / server_layers
            layers = [server_layers // gpu_count] * gpu_count
            for index in range(server_layers % gpu_count):
                layers[index] += 1
            per_gpu = tuple(round(per_layer * count, 2) for count in layers)
            return Objective(
                client_peak_mem_alloc=client,
                server_peak_mem_alloc=with_nccl,
                min_gpu_num_required=gpu_count,
                layers_num_per_gpu=tuple(layers),
                mem_alloc_per_gpu=per_gpu,
                mem_alloc_per_layer=per_layer,
            )
    return Objective(client_peak_mem_alloc=client, server_peak_mem_alloc=server, feasible=False)


def _duration(base: float, increment: float, split: int, baseline: int) -> float:
    return base + (split - baseline) * increment


def simulate_splitmind(
    variables: MainVariable,
    timing: TimeCost,
    memory: MemoryCost,
    *,
    link: LinkModel | None = None,
    activation_nbytes: int = 0,
    legacy_compat: bool = False,
) -> SimulationResult:
    """Simulate the paper's fill-drain SplitMind schedule."""
    objective = _simulate_memory(variables, memory)
    if not objective.feasible:
        return SimulationResult(variables, objective)
    microbatches = variables.batch_size // memory.micro_batch_size
    split = variables.split_point
    baseline = memory.baseline_split_point
    gpu_count = objective.min_gpu_num_required
    layers = objective.layers_num_per_gpu
    total_layers = sum(layers)
    head_fwd = _duration(timing.base_head_fwd_time_per_mb, timing.head_fwd_time_increment_per_sp, split, baseline)
    head_bwd = _duration(timing.base_head_bwd_time_per_mb, timing.head_bwd_time_increment_per_sp, split, baseline)
    tail_fwd = _duration(timing.base_tail_fwd_time_per_mb, timing.tail_fwd_time_increment_per_sp, split, baseline)
    tail_bwd = _duration(timing.base_tail_bwd_time_per_mb, timing.tail_bwd_time_increment_per_sp, split, baseline)
    server_fwd_total = (
        timing.base_server_fwd_time_per_mb
        + (memory.max_split_point - split - baseline) * timing.server_fwd_time_increment_per_sp
    )
    server_bwd_total = (
        timing.base_server_bwd_time_per_mb
        + (memory.max_split_point - split - baseline) * timing.server_bwd_time_increment_per_sp
    )
    server_fwd = [server_fwd_total / total_layers * count for count in layers]
    server_bwd = [server_bwd_total / total_layers * count for count in layers]
    model_layers = variables.client_offload_model_state_sp_num
    head_offload = (
        timing.base_head_model_state_offload_time
        + (model_layers - baseline) * timing.head_model_offload_time_increment_per_sp
    )
    tail_offload = (
        timing.base_tail_model_state_offload_time
        + (model_layers - baseline) * timing.tail_model_offload_time_increment_per_sp
    )
    head_reload = head_offload
    tail_reload = tail_offload
    head_activation_offload = _duration(
        timing.base_head_activation_offload_time_per_mb,
        timing.head_activation_offload_time_increment_per_sp,
        split,
        baseline,
    )
    head_activation_reload = _duration(
        timing.base_head_activation_reload_time_per_mb,
        timing.head_activation_reload_time_increment_per_sp,
        split,
        baseline,
    )
    server_activation_offload_total = _duration(
        timing.base_server_activation_offload_time_per_mb,
        timing.server_activation_offload_time_increment_per_sp,
        split,
        baseline,
    )
    server_activation_reload_total = _duration(
        timing.base_server_activation_reload_time_per_mb,
        timing.server_activation_reload_time_increment_per_sp,
        split,
        baseline,
    )
    server_activation_offload = [server_activation_offload_total / total_layers * count for count in layers]
    server_activation_reload = [server_activation_reload_total / total_layers * count for count in layers]
    if link is not None and activation_nbytes:
        transfer = link.transfer_ms(activation_nbytes)
        head_send = server_send = tail_send = server_grad_send = transfer
    else:
        extra = 10.0 if legacy_compat else timing.delay_time_avg_ms
        head_send = timing.head_activation_send_time + extra
        server_send = timing.server_activation_send_time + extra
        tail_send = timing.tail_gradient_send_time + extra
        server_grad_send = timing.server_gradient_send_time + extra

    def intervals(count: int) -> list[list[float]]:
        return [[0.0, 0.0] for _ in range(count)]

    head_fwd_ts = intervals(microbatches)
    head_send_ts = intervals(microbatches)
    tail_fwd_ts = intervals(microbatches)
    tail_bwd_ts = intervals(microbatches)
    tail_send_ts = intervals(microbatches)
    head_bwd_ts = intervals(microbatches)
    server_send_ts = intervals(microbatches)
    server_grad_send_ts = intervals(microbatches)
    server_fwd_ts = [intervals(microbatches) for _ in range(gpu_count)]
    server_bwd_ts = [intervals(microbatches) for _ in range(gpu_count)]

    for index in range(microbatches):
        if index:
            head_fwd_ts[index][0] = head_fwd_ts[index - 1][1]
        head_fwd_ts[index][1] = head_fwd_ts[index][0] + max(
            head_fwd,
            head_activation_offload if index < variables.client_offload_mb_num else 0.0,
        )
    for index in range(microbatches):
        head_send_ts[index][0] = (
            head_fwd_ts[index][1] if index == 0 else max(head_fwd_ts[index][1], head_send_ts[index - 1][1])
        )
        head_send_ts[index][1] = head_send_ts[index][0] + head_send

    for index in range(microbatches):
        for rank in range(gpu_count):
            dependencies = [head_send_ts[index][1] if rank == 0 else server_fwd_ts[rank - 1][index][1]]
            if index:
                dependencies.append(server_fwd_ts[rank][index - 1][1])
            if index > 1:
                previous = index - 2
                offload_limit = variables.client_offload_mb_num if legacy_compat else variables.server_offload_mb_num
                if previous < offload_limit:
                    dependencies.append(server_fwd_ts[rank][previous][1] + server_activation_offload[rank])
            server_fwd_ts[rank][index][0] = max(dependencies)
            server_fwd_ts[rank][index][1] = server_fwd_ts[rank][index][0] + server_fwd[rank]
    for index in range(microbatches):
        server_send_ts[index][0] = (
            server_fwd_ts[-1][index][1]
            if index == 0
            else max(server_fwd_ts[-1][index][1], server_send_ts[index - 1][1])
        )
        server_send_ts[index][1] = server_send_ts[index][0] + server_send

    for index in range(microbatches):
        if index == 0:
            tail_fwd_ts[index][0] = max(
                head_fwd_ts[-1][1] + head_offload,
                head_fwd_ts[-1][1] + tail_reload,
                server_send_ts[index][1],
            ) + (head_activation_reload if variables.client_offload_mb_num else 0.0)
        else:
            tail_fwd_ts[index][0] = max(tail_bwd_ts[index - 1][1], server_send_ts[index][1])
        tail_fwd_ts[index][1] = tail_fwd_ts[index][0] + tail_fwd
        tail_bwd_ts[index][0] = tail_fwd_ts[index][1]
        tail_bwd_ts[index][1] = tail_bwd_ts[index][0] + tail_bwd
    for index in range(microbatches):
        tail_send_ts[index][0] = (
            max(head_send_ts[-1][1], tail_bwd_ts[index][1])
            if index == 0
            else max(tail_send_ts[index - 1][1], tail_bwd_ts[index][1])
        )
        tail_send_ts[index][1] = tail_send_ts[index][0] + tail_send

    for index in range(microbatches):
        for rank in range(gpu_count - 1, -1, -1):
            dependencies = [tail_send_ts[index][1] if rank == gpu_count - 1 else server_bwd_ts[rank + 1][index][1]]
            if index == 0:
                if rank == gpu_count - 1:
                    dependencies.append(server_fwd_ts[rank][-1][1])
                if variables.server_offload_mb_num:
                    dependencies.append(dependencies[0] + server_activation_reload[rank])
            else:
                dependencies.append(server_bwd_ts[rank][index - 1][1])
                if index < variables.server_offload_mb_num:
                    dependencies.append(server_bwd_ts[rank][index - 1][0] + server_activation_reload[rank])
            server_bwd_ts[rank][index][0] = max(dependencies)
            server_bwd_ts[rank][index][1] = server_bwd_ts[rank][index][0] + server_bwd[rank]
    for index in range(microbatches):
        server_grad_send_ts[index][0] = (
            max(server_send_ts[-1][1], server_bwd_ts[0][index][1])
            if index == 0
            else max(server_grad_send_ts[index - 1][1], server_bwd_ts[0][index][1])
        )
        server_grad_send_ts[index][1] = server_grad_send_ts[index][0] + server_grad_send

    for index in range(microbatches):
        if index == 0:
            head_bwd_ts[index][0] = max(
                tail_bwd_ts[-1][1] + tail_offload,
                tail_bwd_ts[-1][1] + head_reload,
                server_grad_send_ts[index][1],
            ) + (head_activation_reload if index < variables.client_offload_mb_num else 0.0)
            head_bwd_ts[index][1] = head_bwd_ts[index][0] + head_bwd
        else:
            head_bwd_ts[index][0] = max(
                head_bwd_ts[index - 1][0] + head_activation_reload,
                head_bwd_ts[index - 1][1],
                server_grad_send_ts[index][1],
            )
            head_bwd_ts[index][1] = head_bwd_ts[index][0] + max(
                head_bwd,
                head_activation_reload if index < variables.client_offload_mb_num else 0.0,
            )

    spans: list[Span] = []

    def add(lane: str, operation: str, index: int, pair: list[float], nbytes: int = 0) -> None:
        spans.append(Span(lane, operation, 0, index, pair[0], pair[1], nbytes))

    for index in range(microbatches):
        add("client", "head_forward", index, head_fwd_ts[index])
        add("uplink", "act_up", index, head_send_ts[index], activation_nbytes)
        add("downlink", "act_down", index, server_send_ts[index], activation_nbytes)
        add("client", "tail_forward", index, tail_fwd_ts[index])
        add("client", "tail_backward", index, tail_bwd_ts[index])
        add("uplink", "grad_up", index, tail_send_ts[index], activation_nbytes)
        add("downlink", "grad_down", index, server_grad_send_ts[index], activation_nbytes)
        add("client", "head_backward", index, head_bwd_ts[index])
        for rank in range(gpu_count):
            add(f"server/{rank}", "server_forward", index, server_fwd_ts[rank][index])
            add(f"server/{rank}", "server_backward", index, server_bwd_ts[rank][index])
    timeline = StepTimeline(0, tuple(spans))
    metrics = summarize([timeline])
    batch_time = head_bwd_ts[-1][1] - head_fwd_ts[0][0]
    epoch_hours = batch_time * variables.total_sample_count / variables.batch_size / 1000 / 3600
    client_compute = sum(
        head_fwd_ts[index][1]
        - head_fwd_ts[index][0]
        + tail_fwd_ts[index][1]
        - tail_fwd_ts[index][0]
        + tail_bwd_ts[index][1]
        - tail_bwd_ts[index][0]
        + head_bwd_ts[index][1]
        - head_bwd_ts[index][0]
        for index in range(microbatches)
    )
    client_send = sum(
        head_send_ts[index][1] - head_send_ts[index][0] + tail_send_ts[index][1] - tail_send_ts[index][0]
        for index in range(microbatches)
    )
    server_send_total = sum(
        server_send_ts[index][1]
        - server_send_ts[index][0]
        + server_grad_send_ts[index][1]
        - server_grad_send_ts[index][0]
        for index in range(microbatches)
    )
    rounded_epoch_hours = round(epoch_hours, 4)
    objective = Objective(
        **{
            **objective.__dict__,
            "batch_train_time": round(batch_time, 4),
            "epoch_train_time": rounded_epoch_hours,
            "server_cost_per_epoch": round(
                objective.min_gpu_num_required * rounded_epoch_hours * variables.gpu_rent_cost_per_hour * 100,
                4,
            ),
            "client_idle_rate": round(100 - client_compute / batch_time * 100, 2),
            "server_idle_rate": 100.0 if legacy_compat else round(metrics.server_idle_rate, 2),
            "client_send_rate": round(client_send / batch_time * 100, 2),
            "server_send_rate": round(server_send_total / batch_time * 100, 2),
        }
    )
    return SimulationResult(variables, objective, timeline, metrics)
