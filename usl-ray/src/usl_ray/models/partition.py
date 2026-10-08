"""Generic decoder partitioning shared by runtime and simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PretrainedConfig

from usl_ray.models.arch import ArchSpec, architecture_spec, resolve_module
from usl_ray.models.sequence import SequenceSpec, build_sequence_inputs


@dataclass(frozen=True)
class LayerRange:
    """Half-open absolute decoder-layer interval."""

    start: int
    stop: int

    @property
    def count(self) -> int:
        """Return layers in this interval."""
        return self.stop - self.start


@dataclass(frozen=True)
class SplitPlan:
    """Symmetric U-shaped model split."""

    num_layers: int
    split_point: int

    def __post_init__(self) -> None:
        if self.split_point < 1 or 2 * self.split_point >= self.num_layers:
            raise ValueError("split_point must leave at least one decoder layer for the server")

    @property
    def head(self) -> LayerRange:
        """Return the client-head interval."""
        return LayerRange(0, self.split_point)

    @property
    def trunk(self) -> LayerRange:
        """Return the server interval."""
        return LayerRange(self.split_point, self.num_layers - self.split_point)

    @property
    def tail(self) -> LayerRange:
        """Return the client-tail interval."""
        return LayerRange(self.num_layers - self.split_point, self.num_layers)

    def server_stages(self, world_size: int) -> tuple[LayerRange, ...]:
        """Split the trunk into balanced, consecutive stage intervals."""
        return balanced_partition(self.trunk, world_size)


def balanced_partition(layer_range: LayerRange, partitions: int) -> tuple[LayerRange, ...]:
    """Partition layers evenly, assigning a remainder to earlier stages."""
    if partitions <= 0 or partitions > layer_range.count:
        raise ValueError("partitions must be positive and no larger than the layer count")
    base, remainder = divmod(layer_range.count, partitions)
    result: list[LayerRange] = []
    start = layer_range.start
    for rank in range(partitions):
        stop = start + base + (1 if rank < remainder else 0)
        result.append(LayerRange(start, stop))
        start = stop
    return tuple(result)


class DecoderPartition(nn.Module):
    """A consecutive set of decoder layers with optional model boundaries."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_range: LayerRange,
        layers: list[nn.Module],
        rotary_embedding: nn.Module,
        *,
        architecture: ArchSpec,
        embedding: nn.Module | None = None,
        norm: nn.Module | None = None,
        lm_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if len(layers) != layer_range.count:
            raise ValueError(f"Expected {layer_range.count} layers, got {len(layers)}")
        self.config = config
        self.layer_range = layer_range
        self.architecture = architecture
        self.embedding = embedding
        self.layers = nn.ModuleList(layers)
        self.rotary_embedding = rotary_embedding
        self.norm = norm
        self.lm_head = lm_head

    @property
    def is_head(self) -> bool:
        """Return whether this partition accepts token IDs."""
        return self.embedding is not None

    @property
    def is_tail(self) -> bool:
        """Return whether this partition emits logits."""
        return self.lm_head is not None

    def layer_param_groups(self) -> dict[int, list[nn.Parameter]]:
        """Map absolute decoder-layer indexes to their parameters."""
        return {
            absolute: list(layer.parameters())
            for absolute, layer in zip(range(self.layer_range.start, self.layer_range.stop), self.layers, strict=True)
        }

    def forward(self, inputs: torch.Tensor, sequence: SequenceSpec) -> torch.Tensor:
        """Run this partition and regenerate all sequence tensors locally."""
        hidden = self.embedding(inputs) if self.embedding is not None else inputs
        if self.embedding is not None and self.architecture.embed_scale == "sqrt_hidden":
            hidden = hidden * torch.tensor(math.sqrt(self.config.hidden_size), dtype=hidden.dtype, device=hidden.device)
        sequence_inputs = build_sequence_inputs(self.config, self.rotary_embedding, hidden, sequence)
        cache_position = torch.arange(
            sequence.position.offset,
            sequence.position.offset + sequence.seq_len,
            device=hidden.device,
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=sequence_inputs.mask_for(layer),
                position_ids=sequence_inputs.position_ids,
                position_embeddings=sequence_inputs.position_embeddings,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
            )
        if self.norm is not None:
            hidden = self.norm(hidden)
        if self.lm_head is None:
            return hidden
        logits = self.lm_head(hidden)
        softcap_attr = self.architecture.final_logit_softcap_attr
        if softcap_attr and (softcap := getattr(self.config, softcap_attr, None)) is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits.float()


def partition_model(
    model: nn.Module,
    architecture: str,
    layer_range: LayerRange,
    *,
    include_embedding: bool = False,
    include_output: bool = False,
) -> DecoderPartition:
    """Build one logical partition while preserving original layer indexes."""
    spec = architecture_spec(architecture)
    all_layers = resolve_module(model, spec.layers)
    if not isinstance(all_layers, nn.ModuleList):
        raise TypeError(f"{spec.layers} must resolve to nn.ModuleList")
    selected = list(all_layers[layer_range.start : layer_range.stop])
    return DecoderPartition(
        model.config,
        layer_range,
        selected,
        resolve_module(model, spec.rotary),
        architecture=spec,
        embedding=resolve_module(model, spec.embed) if include_embedding else None,
        norm=resolve_module(model, spec.norm) if include_output else None,
        lm_head=resolve_module(model, spec.lm_head) if include_output else None,
    )


def partition_u_shape(
    model: nn.Module,
    architecture: str,
    split_point: int,
    server_world_size: int,
) -> tuple[DecoderPartition, tuple[DecoderPartition, ...], DecoderPartition]:
    """Build client head, balanced server stages, and client tail."""
    plan = SplitPlan(int(model.config.num_hidden_layers), split_point)
    head = partition_model(model, architecture, plan.head, include_embedding=True)
    stages = tuple(partition_model(model, architecture, item) for item in plan.server_stages(server_world_size))
    tail = partition_model(model, architecture, plan.tail, include_output=True)
    return head, stages, tail
