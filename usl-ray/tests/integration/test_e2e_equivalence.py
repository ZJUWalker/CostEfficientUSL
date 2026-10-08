"""One-step numerical equivalence of Ray USL and an unsplit autograd graph."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import ray
import torch
import torch.nn.functional as functional
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config

from usl_ray.client.trainer import ClientActor
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
from usl_ray.data import SyntheticDataset
from usl_ray.models.partition import partition_u_shape
from usl_ray.models.sequence import SequenceSpec
from usl_ray.server.launcher import create_stage_actors


def _server_config(path: Path, mode: str, world_size: int) -> ExperimentConfig:
    microbatch = 2 if mode == "sequential" else 1
    return ExperimentConfig(
        run=RunConfig(name=f"equivalence-{mode}", seed=17),
        model=ModelConfig(
            path=str(path),
            init="random",
            architecture="qwen3",
            num_hidden_layers=4,
            hidden_size=16,
            vocab_size=32,
            dtype="float32",
            attn_implementation="eager",
        ),
        split=SplitConfig(split_point=1),
        batch=BatchConfig(batch_size=2, micro_batch_size=microbatch, max_seq_len=4),
        train=TrainConfig(steps=1, learning_rate=1e-3, max_grad_norm=1e9),
        pipeline=PipelineConfig(mode=mode),
        profile=ProfileConfig(enabled=False, warmup_steps=0),
        server=ServerConfig(world_size=world_size, backend="gloo", wait_client_s=30),
    )


def _reference(config: ExperimentConfig) -> tuple[float, tuple[dict[str, torch.Tensor], ...]]:
    torch.manual_seed(config.run.seed)
    hf_config = AutoConfig.from_pretrained(config.model.path, local_files_only=True)
    hf_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(hf_config, dtype=torch.float32, attn_implementation="eager")
    world_size = config.server.world_size if config.server else 1
    head, stages, tail = partition_u_shape(model, "qwen3", 1, world_size)
    modules = (head, *stages, tail)
    for module in modules:
        module.train()
    head_ids = {id(parameter) for parameter in head.parameters()}
    optimizers = [
        torch.optim.AdamW(head.parameters(), lr=config.train.learning_rate),
        *[torch.optim.AdamW(stage.parameters(), lr=config.train.learning_rate) for stage in stages],
        torch.optim.AdamW(
            [parameter for parameter in tail.parameters() if id(parameter) not in head_ids],
            lr=config.train.learning_rate,
        ),
    ]
    dataset = SyntheticDataset(2, 4, 32, config.run.seed)
    input_ids = torch.stack([dataset[index]["input_ids"] for index in range(2)])
    sequence = SequenceSpec(4)
    losses: list[torch.Tensor] = []
    for ids in torch.split(input_ids, config.batch.micro_batch_size):
        hidden = head(ids, sequence)
        for stage in stages:
            hidden = stage(hidden, sequence)
        logits = tail(hidden, sequence)
        losses.append(
            functional.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.shape[-1]),
                ids[..., 1:].reshape(-1),
                reduction="sum",
            )
            / (input_ids.shape[0] * (input_ids.shape[1] - 1))
        )
    loss = sum(losses)
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    return float(loss.detach()), tuple(
        {name: value.detach().clone() for name, value in module.state_dict().items()} for module in modules
    )


@pytest.mark.parametrize("mode", ["sequential", "gpipe", "splitmind", "splitmind_eager", "1f1b"])
def test_one_step_matches_connected_graph(tmp_path: Path, mode: str) -> None:
    """All client schedules produce the same loss and parameter update."""
    Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        attention_dropout=0.0,
    ).save_pretrained(tmp_path)
    world_size = 1 if mode == "sequential" else 2
    server_config = _server_config(tmp_path, mode, world_size)
    client_config = replace(
        server_config,
        server=None,
        client=ClientConfig(data=DataConfig(dataset="synthetic", sample_count=2)),
    )
    expected_loss, expected_states = _reference(server_config)
    ray.init(
        address="local",
        num_cpus=8,
        resources={"usl_server": 2, "usl_client": 1},
        include_dashboard=False,
    )
    stages: list[ray.actor.ActorHandle] = []
    group = None
    client = None
    try:
        stages, group = create_stage_actors(server_config)
        client = ClientActor.options(num_cpus=1, resources={"usl_client": 1}).remote(client_config, stages[0])
        server_info = ray.get(stages[0].attach.remote(client, client_config))
        ray.get(client.configure.remote(server_info))
        server_runs = [stage.run.remote() for stage in stages]
        client_run = client.train.remote()
        server_results = ray.get(server_runs)
        ray.get(stages[0].publish_profile.remote(server_results))
        result = ray.get(client_run)
        head_state, tail_state = ray.get(client.parameter_state.remote())
        stage_states = ray.get([stage.parameter_state.remote() for stage in stages])
        actual_states = (head_state, *stage_states, tail_state)
        assert result.losses == pytest.approx((expected_loss,), abs=1e-5)
        for actual, expected in zip(actual_states, expected_states, strict=True):
            assert actual.keys() == expected.keys()
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-5)
    finally:
        if client is not None:
            ray.kill(client, no_restart=True)
        for stage in stages:
            ray.kill(stage, no_restart=True)
        if group is not None:
            from ray.util.placement_group import remove_placement_group

            remove_placement_group(group)
        ray.shutdown()
