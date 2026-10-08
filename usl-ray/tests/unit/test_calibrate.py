"""Cost-model calibration validation."""

from __future__ import annotations

import pytest

from usl_ray.simulate.cost_model import fit_from_profiles


def _profile(split: int, batch: int, client_memory: float, server_memory: float) -> dict:
    scale = split / 2
    return {
        "config": {
            "model": {"hidden_size": 128},
            "network": {"mbps": 230, "delay_ms": 10},
        },
        "client_max_memory_mib": client_memory,
        "server_max_memory_mib": [server_memory],
        "metrics": {
            "operation_mean_ms": {
                "head_forward": 10 * scale,
                "head_backward": 20 * scale,
                "tail_forward": 11 * scale,
                "tail_backward": 21 * scale,
                "server_forward": 60 / scale,
                "server_backward": 90 / scale,
                "act_up": 30,
                "act_down": 30,
                "grad_up": 30,
                "grad_down": 30,
                "head_model_offload": split,
                "tail_model_offload": split + 1,
            }
        },
    }


def test_calibration_lists_missing_combinations() -> None:
    """Partial calibration input fails with actionable missing keys."""
    with pytest.raises(ValueError, match="Missing calibration combinations"):
        fit_from_profiles(
            {},
            base_split_point=2,
            base_batch_size=8,
            max_split_point=14,
            micro_batch_size=1,
            max_seq_len=128,
            lora=True,
        )


def test_seven_profiles_fit_a_complete_cost_model() -> None:
    """All legacy difference fields are populated from the seven profiles."""
    keys = [
        (2, 8, 0, 0, 0),
        (2, 8, 8, 0, 8),
        (3, 8, 0, 0, 0),
        (3, 8, 8, 0, 8),
        (2, 8, 0, 2, 0),
        (3, 8, 0, 3, 0),
        (2, 9, 9, 0, 9),
    ]
    profiles = {
        key: _profile(
            key[0], key[1], 3000 + key[0] * 100 - key[2] * 10 - key[3] * 50, 9000 - key[0] * 100 - key[4] * 10
        )
        for key in keys
    }
    model = fit_from_profiles(
        profiles,
        base_split_point=2,
        base_batch_size=8,
        max_split_point=14,
        micro_batch_size=1,
        max_seq_len=128,
        lora=True,
    )
    assert model.time.rate_mbps == 230
    assert model.time.head_activation_send_time == 30
    assert model.memory.mem_increment_per_sp_client == 100
    assert model.memory.base_model_state_mem_alloc_client == 100
