"""Readable client pipeline schedules expressed with bounded loops and futures."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, wait
from typing import Protocol, TypeVar

T = TypeVar("T")


class TrainerProtocol(Protocol[T]):
    """Operations required by all client schedules."""

    server_world_size: int

    def head_forward(self, microbatch: T) -> None: ...

    def activation(self, microbatch: T) -> Future: ...

    def tail_forward(self, microbatch: T, activation: object) -> None: ...

    def tail_backward(self, microbatch: T) -> float: ...

    def gradient(self, microbatch: T) -> Future: ...

    def head_backward(self, microbatch: T, gradient: object) -> None: ...

    def swap_to_tail(self) -> None: ...

    def swap_to_head(self) -> None: ...

    def step_tail(self) -> None: ...

    def step_head(self) -> None: ...


def run_sequential(trainer: TrainerProtocol[T], microbatches: list[T]) -> float:
    """Run one complete U-shaped path before starting another."""
    total = 0.0
    for microbatch in microbatches:
        trainer.head_forward(microbatch)
        trainer.tail_forward(microbatch, trainer.activation(microbatch).result())
        total += trainer.tail_backward(microbatch)
        trainer.head_backward(microbatch, trainer.gradient(microbatch).result())
    trainer.step_tail()
    trainer.step_head()
    return total


def run_gpipe(trainer: TrainerProtocol[T], microbatches: list[T]) -> float:
    """Run fill-drain on both client model partitions."""
    for microbatch in microbatches:
        trainer.head_forward(microbatch)
    trainer.swap_to_tail()
    for microbatch in microbatches:
        trainer.tail_forward(microbatch, trainer.activation(microbatch).result())
    total = sum(trainer.tail_backward(microbatch) for microbatch in microbatches)
    trainer.step_tail()
    trainer.swap_to_head()
    for microbatch in microbatches:
        trainer.head_backward(microbatch, trainer.gradient(microbatch).result())
    trainer.step_head()
    return total


def run_splitmind(trainer: TrainerProtocol[T], microbatches: list[T]) -> float:
    """Run all head forward, tail forward/backward, then head backward."""
    for microbatch in microbatches:
        trainer.head_forward(microbatch)
    trainer.swap_to_tail()
    total = 0.0
    for microbatch in microbatches:
        trainer.tail_forward(microbatch, trainer.activation(microbatch).result())
        total += trainer.tail_backward(microbatch)
    trainer.step_tail()
    trainer.swap_to_head()
    for microbatch in microbatches:
        trainer.head_backward(microbatch, trainer.gradient(microbatch).result())
    trainer.step_head()
    return total


def run_splitmind_eager(trainer: TrainerProtocol[T], microbatches: list[T]) -> float:
    """Process each boundary result as soon as its future is ready."""
    for microbatch in microbatches:
        trainer.head_forward(microbatch)
    trainer.swap_to_tail()
    pending: dict[Future, tuple[str, T]] = {
        trainer.activation(microbatch): ("activation", microbatch) for microbatch in microbatches
    }
    total = 0.0
    for _ in range(2 * len(microbatches)):
        ready, _ = wait(pending, return_when=FIRST_COMPLETED)
        future = next(iter(ready))
        kind, microbatch = pending.pop(future)
        if kind == "activation":
            trainer.tail_forward(microbatch, future.result())
            total += trainer.tail_backward(microbatch)
            pending[trainer.gradient(microbatch)] = ("gradient", microbatch)
        else:
            trainer.head_backward(microbatch, future.result())
    if pending:
        raise RuntimeError(f"Eager schedule left {len(pending)} unfinished operations")
    trainer.step_tail()
    trainer.step_head()
    return total


def run_1f1b(trainer: TrainerProtocol[T], microbatches: list[T]) -> float:
    """Client-side warmup and one-forward/one-backward schedule.

    Torch rank zero needs ``server_world_size`` activations before its first
    backward. Warming up ``S + 2`` therefore cannot deadlock the server.
    """
    warmup = min(len(microbatches), trainer.server_world_size + 2)
    next_forward = 0
    pending: dict[Future, tuple[str, T]] = {}
    for _ in range(warmup):
        microbatch = microbatches[next_forward]
        trainer.head_forward(microbatch)
        pending[trainer.activation(microbatch)] = ("activation", microbatch)
        next_forward += 1
    total = 0.0
    for _ in range(2 * len(microbatches)):
        ready, _ = wait(pending, return_when=FIRST_COMPLETED)
        future = next(iter(ready))
        kind, microbatch = pending.pop(future)
        if kind == "activation":
            trainer.tail_forward(microbatch, future.result())
            total += trainer.tail_backward(microbatch)
            pending[trainer.gradient(microbatch)] = ("gradient", microbatch)
        else:
            trainer.head_backward(microbatch, future.result())
            if next_forward < len(microbatches):
                following = microbatches[next_forward]
                trainer.head_forward(following)
                pending[trainer.activation(following)] = ("activation", following)
                next_forward += 1
    if pending or next_forward != len(microbatches):
        raise RuntimeError("1F1B schedule did not consume every microbatch")
    trainer.step_tail()
    trainer.step_head()
    return total


SCHEDULES = {
    "sequential": run_sequential,
    "gpipe": run_gpipe,
    "splitmind": run_splitmind,
    "splitmind_eager": run_splitmind_eager,
    "1f1b": run_1f1b,
}
