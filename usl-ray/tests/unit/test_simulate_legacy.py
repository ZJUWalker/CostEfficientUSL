"""Compatibility fixture for the deterministic legacy SplitMind simulator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from usl_ray.simulate.cost_model import MainVariable, MemoryCost, TimeCost
from usl_ray.simulate.simulator import simulate_splitmind


def test_legacy_objective_is_reproduced() -> None:
    """The port preserves legacy arithmetic when explicit compatibility is requested."""
    path = Path(__file__).parents[1] / "fixtures" / "legacy_simulate" / "splitmind.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    result = simulate_splitmind(
        MainVariable(**fixture["main_variable"]),
        TimeCost(**fixture["timing"]),
        MemoryCost(**fixture["memory"]),
        legacy_compat=True,
    )
    expected = fixture["objective"]
    actual = result.objective
    for key in (
        "client_peak_mem_alloc",
        "server_peak_mem_alloc",
        "batch_train_time",
        "server_cost_per_epoch",
        "mem_alloc_per_layer",
        "epoch_train_time",
        "client_idle_rate",
        "server_idle_rate",
        "client_send_rate",
        "server_send_rate",
    ):
        assert getattr(actual, key) == pytest.approx(expected[key], abs=1e-6), key
    assert list(actual.layers_num_per_gpu) == expected["layers_num_per_gpu"]
    assert list(actual.mem_alloc_per_gpu) == expected["mem_alloc_per_gpu"]
