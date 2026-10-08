"""Profile comparison and small Pareto search."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from usl_ray.profile.timeline import StepMetrics
from usl_ray.simulate.cost_model import CostModel, MainVariable
from usl_ray.simulate.simulator import SimulationResult, simulate_splitmind


@dataclass(frozen=True)
class Comparison:
    """Measured versus simulated objective errors."""

    simulated_batch_ms: float
    measured_batch_ms: float
    batch_error_rate: float
    simulated_client_memory_mib: float
    measured_client_memory_mib: float
    client_memory_error_rate: float
    simulated_server_memory_mib: float
    measured_server_memory_mib: float
    server_memory_error_rate: float


def _error_rate(simulated: float, measured: float) -> float:
    return 0.0 if measured == 0 else (simulated - measured) / measured * 100


def compare_profile(path: str | Path, cost_model: CostModel) -> Comparison:
    """Simulate the configuration embedded in one runtime profile."""
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    config = profile["config"]
    client = config["client"]
    variables = MainVariable(
        total_sample_count=client["data"]["sample_count"],
        batch_size=config["batch"]["batch_size"],
        split_point=config["split"]["split_point"],
        client_offload_mb_num=client["activation_offload_mb"],
        server_offload_mb_num=config["server"]["activation_offload_mb"],
        client_offload_model_state_sp_num=client["model_state_offload_layers"],
        lora=config["model"]["lora"]["enabled"],
    )
    result = simulate_splitmind(variables, cost_model.time, cost_model.memory)
    metrics = StepMetrics(**profile["metrics"])
    measured_client = float(profile["client_max_memory_mib"])
    measured_server = max(float(item) for item in profile["server_max_memory_mib"])
    return Comparison(
        result.objective.batch_train_time,
        metrics.batch_train_time_ms,
        _error_rate(result.objective.batch_train_time, metrics.batch_train_time_ms),
        result.objective.client_peak_mem_alloc,
        measured_client,
        _error_rate(result.objective.client_peak_mem_alloc, measured_client),
        result.objective.server_peak_mem_alloc,
        measured_server,
        _error_rate(result.objective.server_peak_mem_alloc, measured_server),
    )


def pareto_search(
    base: MainVariable,
    cost_model: CostModel,
    *,
    max_batch_size: int,
    max_client_memory_mib: float,
) -> list[SimulationResult]:
    """Enumerate readable search variables and return non-dominated results."""
    candidates: list[SimulationResult] = []
    for batch_size in range(cost_model.memory.baseline_minibatch_num, max_batch_size + 1):
        for split_point in range(1, cost_model.memory.max_split_point):
            variables = replace(base, batch_size=batch_size, split_point=split_point)
            result = simulate_splitmind(variables, cost_model.time, cost_model.memory)
            if result.objective.feasible and result.objective.client_peak_mem_alloc <= max_client_memory_mib:
                candidates.append(result)
    front: list[SimulationResult] = []
    for candidate in sorted(candidates, key=lambda item: item.objective.server_cost_per_epoch):
        if any(
            existing.objective.server_cost_per_epoch <= candidate.objective.server_cost_per_epoch
            and existing.objective.epoch_train_time <= candidate.objective.epoch_train_time
            for existing in front
        ):
            continue
        front = [
            existing
            for existing in front
            if not (
                candidate.objective.server_cost_per_epoch <= existing.objective.server_cost_per_epoch
                and candidate.objective.epoch_train_time <= existing.objective.epoch_train_time
            )
        ]
        front.append(candidate)
    return sorted(front, key=lambda item: (item.objective.server_cost_per_epoch, item.objective.epoch_train_time))


def save_search(results: list[SimulationResult], path: str | Path) -> Path:
    """Persist a Pareto front as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            [{"variables": asdict(item.variables), "objective": asdict(item.objective)} for item in results],
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination
