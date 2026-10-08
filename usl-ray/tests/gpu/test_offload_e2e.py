"""CUDA correctness tests for activation and model-state offload."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from usl_ray.models.partition import partition_u_shape
from usl_ray.models.sequence import SequenceSpec
from usl_ray.offload import AsyncActivationOffloader, ModelStateSwapper

pytestmark = pytest.mark.gpu


def _head() -> torch.nn.Module:
    model = Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            attention_dropout=0.0,
        )
    )
    head, _, _ = partition_u_shape(model, "qwen3", 1, 1)
    return head.cuda().train()


def test_model_state_swap_preserves_parameter_update() -> None:
    """Saved parameter views survive a full D2H/H2D layer swap."""
    torch.manual_seed(9)
    reference = _head()
    actual = copy.deepcopy(reference)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    actual_optimizer = torch.optim.AdamW(actual.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 64, (2, 8), device="cuda")
    sequence = SequenceSpec(8)

    reference(input_ids, sequence).sum().backward()
    reference_optimizer.step()

    swapper = ModelStateSwapper(actual, actual_optimizer, 1, torch.device("cuda"))
    hooks = AsyncActivationOffloader(torch.device("cuda"), swapper.parameters)
    with hooks.context(offload_activations=False):
        output = actual(input_ids, sequence)
    assert swapper.bytes > 0
    swapper.offload()
    assert all(parameter.device.type == "cpu" and parameter.numel() == 0 for parameter in swapper.parameters)
    swapper.reload()
    output.sum().backward()
    actual_optimizer.step()
    hooks.close()

    for actual_value, expected_value in zip(actual.state_dict().values(), reference.state_dict().values(), strict=True):
        torch.testing.assert_close(actual_value, expected_value, atol=1e-5, rtol=1e-5)


def test_activation_offload_preserves_gradients() -> None:
    """Saved activation D2H/H2D does not alter model or input gradients."""
    torch.manual_seed(11)
    reference = _head()
    actual = copy.deepcopy(reference)
    reference_input = torch.randint(0, 64, (2, 8), device="cuda")
    actual_input = reference_input.clone()
    sequence = SequenceSpec(8)
    reference(reference_input, sequence).square().mean().backward()
    offloader = AsyncActivationOffloader(torch.device("cuda"))
    with offloader.context(offload_activations=True):
        actual(actual_input, sequence).square().mean().backward()
    offloader.close()
    for actual_parameter, expected_parameter in zip(actual.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, atol=1e-5, rtol=1e-5)
