"""Create and supervise server pipeline actors."""

from __future__ import annotations

from collections.abc import Sequence

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from usl_ray.config import ExperimentConfig, shared_config_digest
from usl_ray.server.stage_actor import StageActor, StageResult

RAY_NAMESPACE = "usl-ray"


def stage_zero_name(config: ExperimentConfig) -> str:
    """Return the stable actor name discovered by the client process."""
    return f"usl-stage0-{shared_config_digest(config)[:16]}"


def create_stage_actors(config: ExperimentConfig) -> tuple[list[ray.actor.ActorHandle], object]:
    """Allocate a strict server-node placement group and initialize all ranks."""
    if config.server is None:
        raise ValueError("server section is required")
    gpu = 1 if config.server.backend == "nccl" else 0
    bundle = {"CPU": 1, "usl_server": 1}
    if gpu:
        bundle["GPU"] = gpu
    bundles = [dict(bundle) for _ in range(config.server.world_size)]
    group = placement_group(bundles, strategy="STRICT_PACK")
    ray.get(group.ready(), timeout=config.server.wait_client_s)
    stages: list[ray.actor.ActorHandle] = []
    for rank in range(config.server.world_size):
        options: dict[str, object] = {
            "num_cpus": 1,
            "num_gpus": gpu,
            "resources": {"usl_server": 1},
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=group,
                placement_group_bundle_index=rank,
            ),
        }
        if rank == 0:
            options["name"] = stage_zero_name(config)
        stages.append(StageActor.options(**options).remote(config, rank))
    ray.get([stage.set_peers.remote(stages) for stage in stages])
    gpu_ids = [ids[0] for ids in ray.get([stage.gpu_ids.remote() for stage in stages]) if ids]
    endpoint = ray.get(stages[0].distributed_endpoint.remote())
    ray.get([stage.setup.remote(gpu_ids, endpoint) for stage in stages])
    return stages, group


def run_server(config: ExperimentConfig) -> list[StageResult]:
    """Run all server stages and release every Ray resource on completion."""
    stages, group = create_stage_actors(config)
    try:
        results = ray.get([stage.run.remote() for stage in stages])
        ray.get(stages[0].publish_profile.remote(results))
        return results
    except BaseException as error:
        refs = [stage.abort.remote(f"Server launcher failed: {error}") for stage in stages]
        ray.wait(refs, num_returns=len(refs), timeout=2)
        raise
    finally:
        for stage in stages:
            ray.kill(stage, no_restart=True)
        remove_placement_group(group)


def stop_stages(stages: Sequence[ray.actor.ActorHandle], reason: str) -> None:
    """Best-effort shutdown helper used by command error paths."""
    refs = [stage.abort.remote(reason) for stage in stages]
    ray.wait(refs, num_returns=len(refs), timeout=2)
