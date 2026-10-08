"""Reconstruct attention masks and positional inputs from compact metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.masking_utils import (
    create_bidirectional_mask,
    create_causal_mask,
    create_sliding_window_causal_mask,
)


@dataclass(frozen=True)
class MaskSpec:
    """Attention relation requested by one microbatch."""

    kind: Literal["causal", "full"] = "causal"


@dataclass(frozen=True)
class PositionSpec:
    """Rule used to reconstruct position IDs on every partition."""

    kind: Literal["arange", "from_valid_lens"] = "arange"
    offset: int = 0


@dataclass(frozen=True)
class SequenceSpec:
    """Tensor-free sequence metadata crossing the client/server boundary."""

    seq_len: int
    valid_lens: tuple[int, ...] | None = None
    mask: MaskSpec = field(default_factory=MaskSpec)
    position: PositionSpec = field(default_factory=PositionSpec)


@dataclass(frozen=True)
class SequenceInputs:
    """Inputs regenerated locally for decoder layers."""

    masks: dict[str, torch.Tensor | None]
    position_ids: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]

    def mask_for(self, layer: nn.Module) -> torch.Tensor | None:
        """Return the mask selected by a decoder layer's attention type."""
        attention_type = getattr(layer, "attention_type", "full_attention")
        return self.masks.get(attention_type, self.masks["full_attention"])


def _padding_mask(hidden: torch.Tensor, spec: SequenceSpec) -> torch.Tensor | None:
    if spec.valid_lens is None:
        return None
    if len(spec.valid_lens) != hidden.shape[0]:
        raise ValueError(f"valid_lens has {len(spec.valid_lens)} items for batch size {hidden.shape[0]}")
    lengths = torch.tensor(spec.valid_lens, dtype=torch.long, device=hidden.device)
    if torch.any(lengths < 0) or torch.any(lengths > spec.seq_len):
        raise ValueError(f"valid_lens values must be in [0, {spec.seq_len}]")
    indexes = torch.arange(spec.seq_len, device=hidden.device).unsqueeze(0)
    if spec.position.kind == "from_valid_lens":
        return (indexes >= spec.seq_len - lengths.unsqueeze(1)).long()
    return (indexes < lengths.unsqueeze(1)).long()


def _position_ids(hidden: torch.Tensor, padding_mask: torch.Tensor | None, spec: SequenceSpec) -> torch.Tensor:
    if spec.position.kind == "arange":
        return (
            torch.arange(spec.position.offset, spec.position.offset + spec.seq_len, device=hidden.device)
            .unsqueeze(0)
            .expand(hidden.shape[0], -1)
        )
    if padding_mask is None:
        raise ValueError("Position kind 'from_valid_lens' requires valid_lens")
    positions = padding_mask.cumsum(dim=-1) - 1 + spec.position.offset
    return positions.masked_fill(padding_mask == 0, spec.position.offset)


def build_sequence_inputs(
    config: PretrainedConfig,
    rotary_embedding: nn.Module,
    hidden: torch.Tensor,
    spec: SequenceSpec,
) -> SequenceInputs:
    """Generate mask, position IDs, and RoPE values on the current device."""
    if hidden.ndim != 3:
        raise ValueError(f"hidden must have shape [batch, sequence, hidden], got {tuple(hidden.shape)}")
    if hidden.shape[1] != spec.seq_len:
        raise ValueError(f"SequenceSpec length {spec.seq_len} does not match tensor length {hidden.shape[1]}")
    padding_mask = _padding_mask(hidden, spec)
    position_ids = _position_ids(hidden, padding_mask, spec)
    cache_position = torch.arange(
        spec.position.offset,
        spec.position.offset + spec.seq_len,
        device=hidden.device,
    )
    if spec.mask.kind == "full":
        full_mask = create_bidirectional_mask(config, hidden, padding_mask)
        masks = {"full_attention": full_mask, "sliding_attention": full_mask}
    else:
        kwargs = {
            "config": config,
            "input_embeds": hidden,
            "attention_mask": padding_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        masks = {"full_attention": create_causal_mask(**kwargs)}
        if "sliding_attention" in getattr(config, "layer_types", []):
            masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
    position_embeddings = rotary_embedding(hidden, position_ids)
    return SequenceInputs(masks=masks, position_ids=position_ids, position_embeddings=position_embeddings)
