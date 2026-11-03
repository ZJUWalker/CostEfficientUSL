"""Ray Actor + stock torch pipeline smoke test over gloo."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import ray
import torch
from transformers import Qwen3Config

from usl_ray.config import (
    BatchConfig,
    ClientConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    ProfileConfig,
    RunConfig,
    ServerConfig,
    SplitConfig,
    TrainConfig,
)
from usl_ray.models.sequence import SequenceSpec
from usl_ray.server.launcher import create_stage_actors
from usl_ray.transport import Mailbox, MessageKey, TensorBlob, clock_ms


@ray.remote(max_concurrency=8)
class FakeClient:
    """Mailbox-only client used to exercise both server boundaries."""

    def __init__(self) -> None:
        self.mailbox = Mailbox()

    def push(self, key: MessageKey, blob: TensorBlob) -> None:
        self.mailbox.put(key, blob)

    def get(self, key: MessageKey) -> TensorBlob:
        return self.mailbox.get(key, 30)

    def abort(self, reason: str) -> None:
        self.mailbox.abort(reason)


def _config(model_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        run=RunConfig(name="gloo-spike"),
        model=ModelConfig(
            path=str(model_path),
            init="random",
            architecture="qwen3",
            num_hidden_layers=4,
            hidden_size=16,
            vocab_size=32,
            dtype="float32",
            attn_implementation="eager",
        ),
        split=SplitConfig(split_point=1),
        batch=BatchConfig(batch_size=2, micro_batch_size=1, max_seq_len=4),
        train=TrainConfig(steps=1),
        profile=ProfileConfig(enabled=True, warmup_steps=0),
        server=ServerConfig(world_size=2, backend="gloo", wait_client_s=30),
    )


@pytest.mark.parametrize("mode", ["splitmind", "1f1b"])
def test_two_stage_stock_pipeline_crosses_ray_boundaries(tmp_path: Path, mode: str) -> None:
    """Two ranks complete forward/backward without custom pipeline scheduling."""
    hf_config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        attention_dropout=0.0,
    )
    hf_config.save_pretrained(tmp_path)
    ray.init(
        address="local",
        num_cpus=6,
        resources={"usl_server": 2, "usl_client": 1},
        include_dashboard=False,
    )
    stages: list[ray.actor.ActorHandle] = []
    group = None
    try:
        server_config = _config(tmp_path)
        server_config = replace(server_config, pipeline=PipelineConfig(mode=mode))
        client_config = replace(
            server_config,
            server=None,
            client=ClientConfig(data=DataConfig(dataset="synthetic", sample_count=2)),
        )
        stages, group = create_stage_actors(server_config)
        client = FakeClient.options(resources={"usl_client": 1}).remote()
        run_refs = [stage.run.remote() for stage in stages]
        ray.get(stages[0].attach.remote(client, client_config))
        ray.get(stages[0].begin_step.remote(0, [SequenceSpec(4), SequenceSpec(4)]))
        for microbatch in range(2):
            activation = torch.randn(1, 4, 16)
            ray.get(
                stages[0].push.remote(
                    MessageKey(0, microbatch, "act_up"),
                    TensorBlob.from_tensor(activation),
                )
            )
        for microbatch in range(2):
            output_blob = ray.get(client.get.remote(MessageKey(0, microbatch, "act_down")))
            gradient = torch.ones(output_blob.shape)
            ray.get(
                stages[-1].push.remote(
                    MessageKey(0, microbatch, "grad_up"),
                    TensorBlob.from_tensor(gradient),
                )
            )
        gradients = [
            ray.get(client.get.remote(MessageKey(0, microbatch, "grad_down"))).to_tensor() for microbatch in range(2)
        ]
        ray.get(stages[0].finish_step.remote(0))
        barrier_returned_ms = clock_ms()
        results = ray.get(run_refs)
        assert [tuple(item.shape) for item in gradients] == [(1, 4, 16), (1, 4, 16)]
        assert [item.rank for item in results] == [0, 1]
        server_step_ends = [
            span.end_ms
            for result in results
            for timeline in result.timelines
            for span in timeline.spans
            if span.operation == "server_step"
        ]
        assert len(server_step_ends) == 2
        assert max(server_step_ends) <= barrier_returned_ms
    finally:
        for stage in stages:
            ray.kill(stage, no_restart=True)
        if group is not None:
            from ray.util.placement_group import remove_placement_group

            remove_placement_group(group)
        ray.shutdown()
