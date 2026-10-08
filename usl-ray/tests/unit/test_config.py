"""Configuration contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from usl_ray.config import ConfigError, diff_shared_config, load_config, same_experiment

CONFIGS = Path(__file__).parents[2] / "experiments" / "configs"


def test_client_and_server_share_one_common_config() -> None:
    """Role-specific files must resolve to identical shared settings."""
    server = load_config(CONFIGS / "smoke-qwen3-0.6b" / "server.yaml", role="server")
    client = load_config(CONFIGS / "smoke-qwen3-0.6b" / "client.yaml", role="client")
    assert same_experiment(server, client)
    assert not diff_shared_config(server, client)
    assert server.server is not None
    assert client.client is not None
    assert server.batch.num_microbatches == 2
    with pytest.raises(FrozenInstanceError):
        server.batch.batch_size = 4  # type: ignore[misc]


def test_typed_cli_overrides() -> None:
    """Overrides preserve dataclass field types and are visible to both roles."""
    config = load_config(
        CONFIGS / "smoke-qwen3-0.6b" / "server.yaml",
        ["batch.batch_size=4", "server.world_size=2", "profile.enabled=false"],
        role="server",
    )
    assert config.batch.batch_size == 4
    assert config.server is not None and config.server.world_size == 2
    assert config.profile.enabled is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["batch.batch_size=3", "batch.micro_batch_size=2"], "divisible"),
        (["split.split_point=14"], "leave at least one"),
        (["pipeline.mode=sequential"], "micro_batch_size == batch_size"),
        (["model.hidden_size=1025", "network.compression=nf4"], "divisible by 64"),
        (["missing.field=1"], "Unknown override"),
    ],
)
def test_invalid_config_is_rejected(overrides: list[str], message: str) -> None:
    """Cross-field invariants fail before actors are created."""
    with pytest.raises(ConfigError, match=message):
        load_config(CONFIGS / "smoke-qwen3-0.6b" / "server.yaml", overrides, role="server")


def test_paper_configs_are_self_contained() -> None:
    """Every paper model config loads without downloading model metadata."""
    for path in sorted((CONFIGS / "paper").glob("*/server.yaml")):
        config = load_config(path, role="server")
        assert config.model.num_hidden_layers > 0
        assert config.server_layer_count > 0
