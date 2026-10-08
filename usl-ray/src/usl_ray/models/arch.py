"""Small declarative registry for supported decoder-only architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import nn


@dataclass(frozen=True)
class ArchSpec:
    """Paths and architecture-specific operations needed by a partition."""

    embed: str = "model.embed_tokens"
    layers: str = "model.layers"
    norm: str = "model.norm"
    lm_head: str = "lm_head"
    rotary: str = "model.rotary_emb"
    embed_scale: Literal["none", "sqrt_hidden"] = "none"
    final_logit_softcap_attr: str | None = None


ARCHS: dict[str, ArchSpec] = {
    "qwen3": ArchSpec(),
    "llama": ArchSpec(),
    "gemma2": ArchSpec(embed_scale="sqrt_hidden", final_logit_softcap_attr="final_logit_softcapping"),
}


def architecture_spec(name: str) -> ArchSpec:
    """Return one supported architecture description."""
    try:
        return ARCHS[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported model architecture {name!r}; choose from {sorted(ARCHS)}") from error


def resolve_module(root: nn.Module, dotted_path: str) -> nn.Module:
    """Resolve a dotted module path without relying on architecture classes."""
    current: object = root
    for component in dotted_path.split("."):
        current = getattr(current, component)
    if not isinstance(current, nn.Module):
        raise TypeError(f"{dotted_path!r} did not resolve to torch.nn.Module")
    return current
