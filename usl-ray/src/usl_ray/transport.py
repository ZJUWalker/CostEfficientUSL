"""Minimal Ray object-store transport for boundary tensors."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import ray
import torch

MessageKind = Literal["act_up", "act_down", "grad_up", "grad_down", "profile"]

_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


@dataclass(frozen=True, order=True)
class MessageKey:
    """Unique boundary message identifier."""

    step: int
    microbatch: int
    kind: MessageKind


@dataclass(frozen=True)
class TensorBlob:
    """A dtype-safe tensor representation backed by a byte ndarray."""

    dtype: str
    shape: tuple[int, ...]
    data: np.ndarray

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorBlob:
        """Create a blob without relying on NumPy bfloat16 support."""
        contiguous = tensor.detach().contiguous()
        if contiguous.device.type != "cpu":
            contiguous = contiguous.cpu()
        dtype_name = str(contiguous.dtype).removeprefix("torch.")
        if dtype_name not in _DTYPES:
            raise TypeError(f"Unsupported tensor dtype: {contiguous.dtype}")
        data = contiguous.view(torch.uint8).numpy()
        return cls(dtype=dtype_name, shape=tuple(contiguous.shape), data=data)

    @property
    def nbytes(self) -> int:
        """Return payload bytes."""
        return int(self.data.nbytes)

    def to_tensor(self, device: torch.device | str = "cpu", *, non_blocking: bool = False) -> torch.Tensor:
        """Restore shape and dtype, optionally copying through pinned CPU memory."""
        try:
            dtype = _DTYPES[self.dtype]
        except KeyError as error:
            raise TypeError(f"Unsupported blob dtype: {self.dtype}") from error
        array = np.asarray(self.data)
        if not array.flags.writeable:
            array = array.copy()
        byte_tensor = torch.from_numpy(array)
        tensor = byte_tensor.view(dtype).reshape(self.shape)
        target = torch.device(device)
        if target.type == "cpu":
            return tensor
        pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
        pinned.copy_(tensor)
        return pinned.to(target, non_blocking=non_blocking)


class Mailbox:
    """Thread-safe keyed futures with abort and timeout semantics."""

    def __init__(self) -> None:
        self._futures: dict[MessageKey, Future[TensorBlob]] = {}
        self._lock = threading.Lock()
        self._abort_reason: str | None = None

    def future(self, key: MessageKey) -> Future[TensorBlob]:
        """Return the unique future for one key."""
        with self._lock:
            future = self._futures.get(key)
            if future is None:
                future = Future()
                if self._abort_reason is not None:
                    future.set_exception(RuntimeError(self._abort_reason))
                self._futures[key] = future
            return future

    def put(self, key: MessageKey, blob: TensorBlob) -> None:
        """Resolve one key exactly once."""
        future = self.future(key)
        if not future.set_running_or_notify_cancel():
            raise RuntimeError(f"Message was cancelled before arrival: {key}")
        try:
            future.set_result(blob)
        except BaseException as error:
            future.set_exception(error)
            raise

    def get(self, key: MessageKey, timeout_s: float) -> TensorBlob:
        """Block for one key and include it in timeout errors."""
        try:
            return self.future(key).result(timeout=timeout_s)
        except TimeoutError as error:
            raise TimeoutError(f"Timed out waiting for {key} after {timeout_s:g}s") from error

    def abort(self, reason: str) -> None:
        """Fail current and future waiters."""
        with self._lock:
            self._abort_reason = reason
            futures = list(self._futures.values())
        for future in futures:
            if not future.done():
                future.set_exception(RuntimeError(reason))

    def discard(self, key: MessageKey) -> None:
        """Release a consumed future."""
        with self._lock:
            self._futures.pop(key, None)


@dataclass
class _StagedTensor:
    cpu_tensor: torch.Tensor
    ready_event: torch.cuda.Event | None
    source: torch.Tensor


class Outbox:
    """FIFO GPU-to-CPU staging followed by Ray put-and-push."""

    def __init__(
        self,
        device: torch.device | str = "cpu",
        on_transfer: Callable[[MessageKey, float, float, int], None] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="usl-outbox")
        self._futures: list[Future[None]] = []
        self._lock = threading.Lock()
        self._stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        self._on_transfer = on_transfer

    def _stage(self, tensor: torch.Tensor, wait_event: torch.cuda.Event | None) -> _StagedTensor:
        source = tensor.detach()
        if source.device.type == "cpu":
            return _StagedTensor(source.contiguous(), None, source)
        if self._stream is None:
            raise RuntimeError("CUDA tensor submitted to a CPU outbox")
        with torch.cuda.stream(self._stream):
            if wait_event is not None:
                self._stream.wait_event(wait_event)
            cpu_tensor = torch.empty_like(source, device="cpu", pin_memory=True)
            cpu_tensor.copy_(source, non_blocking=True)
            ready = torch.cuda.Event()
            self._stream.record_event(ready)
        return _StagedTensor(cpu_tensor, ready, source)

    def _push(self, peer: ray.actor.ActorHandle, key: MessageKey, staged: _StagedTensor) -> None:
        if staged.ready_event is not None:
            staged.ready_event.synchronize()
        started = clock_ms()
        blob = TensorBlob.from_tensor(staged.cpu_tensor)
        blob_ref = ray.put(blob)
        ray.get(peer.push.remote(key, blob_ref))
        if self._on_transfer is not None:
            self._on_transfer(key, started, clock_ms(), blob.nbytes)

    def submit(
        self,
        peer: ray.actor.ActorHandle,
        key: MessageKey,
        tensor: torch.Tensor,
        *,
        wait_event: torch.cuda.Event | None = None,
    ) -> None:
        """Queue one ordered transfer without blocking the compute stream."""
        staged = self._stage(tensor, wait_event)
        future = self._executor.submit(self._push, peer, key, staged)
        with self._lock:
            self._futures.append(future)

    def flush(self) -> None:
        """Wait for every submitted push and surface remote failures."""
        with self._lock:
            futures, self._futures = self._futures, []
        for future in futures:
            future.result()

    def close(self) -> None:
        """Flush pending transfers and stop the worker."""
        self.flush()
        self._executor.shutdown(wait=True)


def clock_ms() -> float:
    """Return a monotonic timestamp suitable for local span durations."""
    return time.perf_counter_ns() / 1_000_000
