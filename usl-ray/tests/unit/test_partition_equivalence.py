"""Numerical equivalence tests for generic decoder partitions."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn
from transformers import Gemma2Config, Gemma2ForCausalLM, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from usl_ray.models.partition import LayerRange, SplitPlan, balanced_partition, partition_u_shape
from usl_ray.models.sequence import SequenceSpec


def _qwen() -> nn.Module:
    return Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=67,
            attention_dropout=0.0,
        )
    )


def _llama() -> nn.Module:
    return LlamaForCausalLM(
        LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=67,
            attention_dropout=0.0,
        )
    )


def _gemma() -> nn.Module:
    return Gemma2ForCausalLM(
        Gemma2Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=67,
            attention_dropout=0.0,
            sliding_window=4,
        )
    )


@pytest.mark.parametrize(
    ("architecture", "factory"),
    [("qwen3", _qwen), ("llama", _llama), ("gemma2", _gemma)],
)
@torch.no_grad()
def test_partition_chain_matches_huggingface_model(architecture: str, factory: Callable[[], nn.Module]) -> None:
    """Head, two trunk stages, and tail reproduce the original model."""
    torch.manual_seed(7)
    model = factory().eval()
    model.config._attn_implementation = "eager"
    head, stages, tail = partition_u_shape(model, architecture, split_point=1, server_world_size=2)
    input_ids = torch.tensor([[3, 4, 5, 6, 0], [7, 8, 9, 10, 11]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    reference = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    sequence = SequenceSpec(seq_len=5, valid_lens=(4, 5))
    hidden = head(input_ids, sequence)
    for stage in stages:
        hidden = stage(hidden, sequence)
    actual = tail(hidden, sequence)
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)


def test_balanced_partition_handles_odd_remainders() -> None:
    """Earlier ranks receive one extra layer and no layer is lost."""
    ranges = balanced_partition(LayerRange(3, 10), 3)
    assert ranges == (LayerRange(3, 6), LayerRange(6, 8), LayerRange(8, 10))
    assert sum(item.count for item in ranges) == 7


def test_layer_parameter_groups_use_absolute_indexes() -> None:
    """Offload code receives explicit absolute indexes instead of parsing names."""
    model = _qwen()
    head, _, _ = partition_u_shape(model, "qwen3", split_point=1, server_world_size=1)
    groups = head.layer_param_groups()
    assert set(groups) == {0}
    assert groups[0]


def test_split_plan_rejects_empty_server() -> None:
    """A split cannot consume every decoder layer on the client."""
    with pytest.raises(ValueError, match="at least one"):
        SplitPlan(num_layers=4, split_point=2)
