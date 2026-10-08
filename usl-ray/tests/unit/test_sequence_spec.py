"""Sequence metadata reconstruction tests."""

from __future__ import annotations

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from usl_ray.models.sequence import MaskSpec, PositionSpec, SequenceSpec, build_sequence_inputs


def _config() -> Qwen3Config:
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
    )
    config._attn_implementation = "eager"
    return config


def test_sequence_spec_contains_no_tensors() -> None:
    """Only scalar metadata crosses a split-learning boundary."""
    spec = SequenceSpec(
        seq_len=5,
        valid_lens=(3, 5),
        mask=MaskSpec("causal"),
        position=PositionSpec("arange"),
    )
    assert not any(isinstance(value, torch.Tensor) for value in vars(spec).values())


def test_right_padding_mask_and_positions_are_rebuilt() -> None:
    """Right-padding metadata reproduces an additive causal mask and positions."""
    config = _config()
    hidden = torch.randn(2, 5, 32)
    inputs = build_sequence_inputs(
        config,
        Qwen3RotaryEmbedding(config),
        hidden,
        SequenceSpec(seq_len=5, valid_lens=(3, 5)),
    )
    assert inputs.position_ids.tolist() == [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    mask = inputs.masks["full_attention"]
    assert mask is not None and mask.shape == (2, 1, 5, 5)
    assert mask[0, 0, 2, 3] < -1e20


def test_left_padding_positions_start_at_zero() -> None:
    """Left-padded valid tokens receive positions starting at zero."""
    config = _config()
    inputs = build_sequence_inputs(
        config,
        Qwen3RotaryEmbedding(config),
        torch.randn(2, 5, 32),
        SequenceSpec(
            seq_len=5,
            valid_lens=(3, 5),
            position=PositionSpec("from_valid_lens"),
        ),
    )
    assert inputs.position_ids.tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]


def test_full_mask_is_bidirectional() -> None:
    """A full mask allows a valid token to attend to later valid tokens."""
    config = _config()
    inputs = build_sequence_inputs(
        config,
        Qwen3RotaryEmbedding(config),
        torch.randn(1, 4, 32),
        SequenceSpec(seq_len=4, valid_lens=(3,), mask=MaskSpec("full")),
    )
    mask = inputs.masks["full_attention"]
    assert mask is not None
    assert mask[0, 0, 0, 2] == 0
    assert mask[0, 0, 0, 3] < -1e20
