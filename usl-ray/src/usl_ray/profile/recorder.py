"""Thread-safe span recording and profile JSON persistence."""

from __future__ import annotations

import json
import platform
import threading
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import ray
import torch
import transformers

from usl_ray.config import ExperimentConfig
from usl_ray.profile.timeline import Span, StepTimeline, summarize, timeline_to_dict
from usl_ray.transport import MessageKey, clock_ms


class Recorder:
    """Collect spans emitted by compute and transport threads."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._spans: list[Span] = []
        self._lock = threading.Lock()

    def add(
        self,
        lane: str,
        operation: str,
        step: int,
        microbatch: int,
        start_ms: float,
        end_ms: float,
        nbytes: int = 0,
    ) -> None:
        """Append one span when profiling is enabled."""
        if not self.enabled:
            return
        span = Span(lane, operation, step, microbatch, start_ms, end_ms, nbytes)
        with self._lock:
            self._spans.append(span)

    def transfer(self, lane: str, key: MessageKey, start_ms: float, end_ms: float, nbytes: int) -> None:
        """Record one acknowledged Ray object transfer."""
        self.add(lane, key.kind, key.step, key.microbatch, start_ms, end_ms, nbytes)

    @contextmanager
    def span(
        self,
        lane: str,
        operation: str,
        step: int,
        microbatch: int,
        device: torch.device,
    ) -> Iterator[None]:
        """Synchronize CUDA around measured computation to match legacy semantics."""
        if not self.enabled:
            yield
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = clock_ms()
        try:
            yield
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.add(lane, operation, step, microbatch, started, clock_ms())

    def timelines(self) -> tuple[StepTimeline, ...]:
        """Group sorted spans by global step."""
        with self._lock:
            spans = list(self._spans)
        steps = sorted({span.step for span in spans if span.step >= 0})
        return tuple(
            StepTimeline(
                step,
                tuple(
                    sorted(
                        (span for span in spans if span.step == step),
                        key=lambda item: (item.start_ms, item.lane, item.operation),
                    )
                ),
            )
            for step in steps
        )


def merge_timelines(*groups: tuple[StepTimeline, ...]) -> tuple[StepTimeline, ...]:
    """Merge client and server timelines by step."""
    steps = sorted({timeline.step for group in groups for timeline in group})
    return tuple(
        StepTimeline(
            step,
            tuple(
                sorted(
                    (
                        span
                        for group in groups
                        for timeline in group
                        if timeline.step == step
                        for span in timeline.spans
                    ),
                    key=lambda item: (item.start_ms, item.lane, item.operation),
                )
            ),
        )
        for step in steps
    )


def write_profile(
    destination: str | Path,
    config: ExperimentConfig,
    timelines: tuple[StepTimeline, ...],
    client_memory_mib: float,
    server_memory_mib: list[float],
) -> Path:
    """Write the current profile contract as JSON."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    measured = timelines[config.profile.warmup_steps :]
    payload = {
        "schema_version": 1,
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "ray": ray.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "client_max_memory_mib": client_memory_mib,
        "server_max_memory_mib": server_memory_mib,
        "timelines": [timeline_to_dict(item) for item in timelines],
        "metrics": asdict(summarize(measured)),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
