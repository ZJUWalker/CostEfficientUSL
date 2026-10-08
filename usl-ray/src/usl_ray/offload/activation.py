"""Asynchronous saved-tensor activation offload with explicit ownership."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch


@dataclass
class _OffloadedTensor:
    cpu: torch.Tensor
    device: torch.device
    ready: torch.cuda.Event
    source: torch.Tensor | None


class _SavedTensorContext(AbstractContextManager[None]):
    def __init__(self, owner: AsyncActivationOffloader, offload_activations: bool) -> None:
        self.owner = owner
        self.offload_activations = offload_activations
        self.hooks: Any = None
        self.previous = False

    def __enter__(self) -> None:
        self.previous = self.owner.offload_activations
        self.owner.offload_activations = self.offload_activations
        self.hooks = torch.autograd.graph.saved_tensors_hooks(self.owner.pack, self.owner.unpack)
        self.hooks.__enter__()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self.hooks is None:
            return None
        try:
            return self.hooks.__exit__(exception_type, exception, traceback)
        finally:
            self.owner.offload_activations = self.previous


class AsyncActivationOffloader:
    """Move tensors saved for backward to pinned CPU memory on a copy stream."""

    def __init__(self, device: torch.device, parameters: list[torch.Tensor] | None = None) -> None:
        if device.type != "cuda":
            raise ValueError("Activation offload requires a CUDA device")
        self.device = device
        self.d2h_stream = torch.cuda.Stream(device)
        self.h2d_stream = torch.cuda.Stream(device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="activation-offload")
        self._states: list[_OffloadedTensor] = []
        self._lock = threading.Lock()
        self.parameter_storages = {
            parameter.untyped_storage().data_ptr() for parameter in (parameters or []) if parameter.numel()
        }
        self.offload_activations = False

    def context(self, *, offload_activations: bool) -> AbstractContextManager[None]:
        """Return a saved-tensor hook context for one forward pass."""
        return _SavedTensorContext(self, offload_activations)

    def _should_offload(self, tensor: torch.Tensor) -> bool:
        if not tensor.is_cuda:
            return False
        if tensor.untyped_storage().data_ptr() in self.parameter_storages:
            return True
        if not self.offload_activations:
            return False
        base = tensor._base
        return base is None or not base.is_leaf

    def pack(self, tensor: torch.Tensor) -> torch.Tensor | _OffloadedTensor:
        """Schedule D2H while retaining source storage until the event completes."""
        if not self._should_offload(tensor):
            return tensor
        current = torch.cuda.current_stream(self.device)
        self.d2h_stream.wait_stream(current)
        with torch.cuda.stream(self.d2h_stream):
            cpu = torch.empty_like(tensor, device="cpu", pin_memory=True)
            cpu.copy_(tensor, non_blocking=True)
            event = torch.cuda.Event()
            self.d2h_stream.record_event(event)
        state = _OffloadedTensor(cpu, tensor.device, event, tensor)
        with self._lock:
            self._states.append(state)
        self.executor.submit(self._release_source, state)
        return state

    @staticmethod
    def _release_source(state: _OffloadedTensor) -> None:
        state.ready.synchronize()
        state.source = None

    def unpack(self, state: torch.Tensor | _OffloadedTensor) -> torch.Tensor:
        """Prefetch one saved tensor and order the compute stream after H2D."""
        if isinstance(state, torch.Tensor):
            return state
        state.ready.synchronize()
        with torch.cuda.stream(self.h2d_stream):
            restored = state.cpu.to(state.device, non_blocking=True)
            ready = torch.cuda.Event()
            self.h2d_stream.record_event(ready)
        torch.cuda.current_stream(state.device).wait_event(ready)
        return restored

    def close(self) -> None:
        """Finish outstanding copies and release CPU backups."""
        self.executor.shutdown(wait=True)
        with self._lock:
            self._states.clear()
