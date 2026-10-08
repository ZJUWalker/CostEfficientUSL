"""Layer-explicit model parameter and optimizer-state swapping."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SwapTimes:
    """Latest measured D2H and H2D durations in milliseconds."""

    offload_ms: float = 0.0
    reload_ms: float = 0.0


def unwrap_layer_groups(model: nn.Module) -> dict[int, list[nn.Parameter]]:
    """Find DecoderPartition's explicit layer groups through an optional PEFT wrapper."""
    candidate: object = model
    for _ in range(4):
        method = getattr(candidate, "layer_param_groups", None)
        if callable(method):
            return method()
        base_model = getattr(candidate, "base_model", None)
        inner = getattr(base_model, "model", None) if base_model is not None else None
        if inner is None:
            break
        candidate = inner
    raise TypeError(f"{type(model).__name__} does not expose layer_param_groups()")


class ModelStateSwapper:
    """Swap selected decoder layers and their optimizer state via pinned memory."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        layer_count: int,
        device: torch.device,
        *,
        excluded_parameter_ids: set[int] | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("Model state swapping requires CUDA")
        groups = unwrap_layer_groups(model)
        selected_layers = sorted(groups)[:layer_count]
        excluded = excluded_parameter_ids or set()
        self.parameters = [
            parameter for layer in selected_layers for parameter in groups[layer] if id(parameter) not in excluded
        ]
        self.optimizer = optimizer
        self.device = device
        self.d2h_stream = torch.cuda.Stream(device)
        self.h2d_stream = torch.cuda.Stream(device)
        self.parameter_cpu: dict[int, torch.Tensor] = {}
        self.optimizer_cpu: dict[tuple[int, str], torch.Tensor] = {}
        self.on_device = True
        self.times = SwapTimes()

    @property
    def bytes(self) -> int:
        """Return selected parameter bytes, excluding optimizer state."""
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters)

    def offload(self) -> None:
        """Synchronously finish an asynchronous D2H batch, then release GPU storage."""
        if not self.on_device or not self.parameters:
            return
        current = torch.cuda.current_stream(self.device)
        self.d2h_stream.wait_stream(current)
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.d2h_stream):
            self.d2h_stream.record_event(start)
            for parameter in self.parameters:
                cpu = self.parameter_cpu.get(id(parameter))
                if cpu is None:
                    cpu = torch.empty_like(parameter, device="cpu", pin_memory=True)
                    self.parameter_cpu[id(parameter)] = cpu
                cpu.copy_(parameter, non_blocking=True)
                for name, value in self.optimizer.state.get(parameter, {}).items():
                    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.is_cuda:
                        key = (id(parameter), name)
                        state_cpu = self.optimizer_cpu.get(key)
                        if state_cpu is None:
                            state_cpu = torch.empty_like(value, device="cpu", pin_memory=True)
                            self.optimizer_cpu[key] = state_cpu
                        state_cpu.copy_(value, non_blocking=True)
            self.d2h_stream.record_event(finish)
        finish.synchronize()
        for parameter in self.parameters:
            parameter.data = torch.empty(0, dtype=parameter.dtype, device="cpu")
            for name, value in self.optimizer.state.get(parameter, {}).items():
                key = (id(parameter), name)
                if key in self.optimizer_cpu and isinstance(value, torch.Tensor):
                    value.data = torch.empty(0, dtype=value.dtype, device="cpu")
        self.on_device = False
        self.times = SwapTimes(start.elapsed_time(finish), self.times.reload_ms)

    def reload(self) -> None:
        """Restore selected layers and optimizer state before computation."""
        if self.on_device or not self.parameters:
            return
        current = torch.cuda.current_stream(self.device)
        self.h2d_stream.wait_stream(current)
        start = torch.cuda.Event(enable_timing=True)
        finish = torch.cuda.Event(enable_timing=True)
        parameter_gpu: dict[int, torch.Tensor] = {}
        optimizer_gpu: dict[tuple[int, str], torch.Tensor] = {}
        with torch.cuda.stream(self.h2d_stream):
            self.h2d_stream.record_event(start)
            for parameter in self.parameters:
                parameter_gpu[id(parameter)] = self.parameter_cpu[id(parameter)].to(self.device, non_blocking=True)
                for name in self.optimizer.state.get(parameter, {}):
                    key = (id(parameter), name)
                    if key in self.optimizer_cpu:
                        optimizer_gpu[key] = self.optimizer_cpu[key].to(self.device, non_blocking=True)
            self.h2d_stream.record_event(finish)
        finish.synchronize()
        for parameter in self.parameters:
            parameter.data = parameter_gpu[id(parameter)]
            for name, value in self.optimizer.state.get(parameter, {}).items():
                key = (id(parameter), name)
                if key in optimizer_gpu and isinstance(value, torch.Tensor):
                    value.data = optimizer_gpu[key]
        self.on_device = True
        self.times = SwapTimes(self.times.offload_ms, start.elapsed_time(finish))
