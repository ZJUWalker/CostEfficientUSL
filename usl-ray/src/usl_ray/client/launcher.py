"""Discover the server and supervise one client actor."""

from __future__ import annotations

import shutil
import threading
from dataclasses import replace

import ray

from usl_ray.client.trainer import ClientActor, ClientResult
from usl_ray.config import ExperimentConfig, dump_config, result_dir
from usl_ray.profile.recorder import write_profile
from usl_ray.server.launcher import RAY_NAMESPACE, stage_zero_name


def wait_for_stage_zero(config: ExperimentConfig) -> ray.actor.ActorHandle:
    """Wait for the named server actor using a bounded retry loop."""
    if config.client is None:
        raise ValueError("client section is required")
    interval = 0.25
    attempts = max(1, int(config.client.wait_server_s / interval))
    last_error: BaseException | None = None
    for _ in range(attempts):
        try:
            actor = ray.get_actor(stage_zero_name(config), namespace=RAY_NAMESPACE)
            ray.get(actor.wait_ready.remote(config.client.wait_server_s))
            return actor
        except ValueError as error:
            last_error = error
            threading.Event().wait(interval)
    raise TimeoutError(f"Server stage zero was not available after {config.client.wait_server_s:g}s") from last_error


def run_client(config: ExperimentConfig, *, num_gpus: float = 1) -> ClientResult:
    """Perform handshake, run the configured steps, and release the client GPU."""
    stage_zero = wait_for_stage_zero(config)
    options = {
        "num_cpus": 1,
        "num_gpus": num_gpus,
        "resources": {"usl_client": 1},
    }
    client = ClientActor.options(**options).remote(config, stage_zero)
    try:
        server_info = ray.get(stage_zero.attach.remote(client, config))
        ray.get(client.configure.remote(server_info))
        result = ray.get(client.train.remote())
        merged = replace(config, server=server_info.server_config)
        output = result_dir(merged)
        if output.exists():
            shutil.rmtree(output)
        dump_config(merged, output / "config.yaml")
        if merged.profile.enabled:
            write_profile(
                output / "profile.json",
                merged,
                result.timelines,
                result.max_memory_mib,
                list(result.server_memory_mib),
            )
        return result
    except BaseException as error:
        stage_zero.abort.remote(f"Client launcher failed: {error}")
        raise
    finally:
        ray.kill(client, no_restart=True)
