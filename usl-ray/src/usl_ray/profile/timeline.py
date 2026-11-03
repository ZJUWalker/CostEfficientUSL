"""Shared profile/simulation timeline contract.

All durations use milliseconds, memory uses MiB, and bandwidth uses SI
megabits per second (1 Mbit/s = 1,000,000 bit/s).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Span:
    """One operation on one logical resource lane."""

    lane: str
    operation: str
    step: int
    microbatch: int
    start_ms: float
    end_ms: float
    nbytes: int = 0

    @property
    def duration_ms(self) -> float:
        """Return non-negative elapsed time."""
        return max(0.0, self.end_ms - self.start_ms)


@dataclass(frozen=True)
class StepTimeline:
    """All spans associated with one global step."""

    step: int
    spans: tuple[Span, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StepMetrics:
    """Derived metrics shared by measured and simulated timelines."""

    batch_train_time_ms: float
    operation_mean_ms: dict[str, float]
    client_idle_rate: float
    server_idle_rate: float
    uplink_bytes: int
    downlink_bytes: int
    uplink_time_ms: float
    downlink_time_ms: float


def shift_timelines(timelines: tuple[StepTimeline, ...], offset_ms: float) -> tuple[StepTimeline, ...]:
    """Translate timeline timestamps into another monotonic clock domain.

    Args:
        timelines: Timelines measured in the source clock domain.
        offset_ms: Value added to every start and end timestamp.

    Returns:
        Timelines expressed in the destination clock domain.
    """
    return tuple(
        StepTimeline(
            timeline.step,
            tuple(
                Span(
                    span.lane,
                    span.operation,
                    span.step,
                    span.microbatch,
                    span.start_ms + offset_ms,
                    span.end_ms + offset_ms,
                    span.nbytes,
                )
                for span in timeline.spans
            ),
        )
        for timeline in timelines
    )


def trimmed_mean(values: list[float]) -> float:
    """Match the legacy profile aggregation: trim extremes when possible."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) > 2:
        ordered = ordered[1:-1]
    return sum(ordered) / len(ordered)


def summarize(timelines: list[StepTimeline] | tuple[StepTimeline, ...]) -> StepMetrics:
    """Derive the same metrics from either runtime or simulation spans."""
    spans = [span for timeline in timelines for span in timeline.spans]
    if not spans:
        return StepMetrics(0.0, {}, 0.0, 0.0, 0, 0, 0.0, 0.0)
    per_step_duration: list[float] = []
    client_idle: list[float] = []
    server_idle: list[float] = []
    for timeline in timelines:
        if not timeline.spans:
            continue
        start = min(span.start_ms for span in timeline.spans)
        end = max(span.end_ms for span in timeline.spans)
        duration = max(0.0, end - start)
        per_step_duration.append(duration)
        if duration:
            client_compute = sum(span.duration_ms for span in timeline.spans if span.lane == "client")
            client_idle.append(max(0.0, 100.0 * (1 - client_compute / duration)))
            server_lanes = sorted({span.lane for span in timeline.spans if span.lane.startswith("server/")})
            if server_lanes:
                rank_idle = [
                    max(
                        0.0,
                        100.0 * (1 - sum(span.duration_ms for span in timeline.spans if span.lane == lane) / duration),
                    )
                    for lane in server_lanes
                ]
                server_idle.append(sum(rank_idle) / len(rank_idle))
    by_operation: dict[str, list[float]] = {}
    for span in spans:
        by_operation.setdefault(span.operation, []).append(span.duration_ms)
    uplink = [span for span in spans if span.lane == "uplink"]
    downlink = [span for span in spans if span.lane == "downlink"]
    return StepMetrics(
        batch_train_time_ms=trimmed_mean(per_step_duration),
        operation_mean_ms={key: trimmed_mean(values) for key, values in sorted(by_operation.items())},
        client_idle_rate=trimmed_mean(client_idle),
        server_idle_rate=trimmed_mean(server_idle),
        uplink_bytes=sum(span.nbytes for span in uplink),
        downlink_bytes=sum(span.nbytes for span in downlink),
        uplink_time_ms=sum(span.duration_ms for span in uplink),
        downlink_time_ms=sum(span.duration_ms for span in downlink),
    )


def timeline_to_dict(timeline: StepTimeline) -> dict[str, Any]:
    """Serialize one timeline without custom JSON encoders."""
    return {"step": timeline.step, "spans": [asdict(span) for span in timeline.spans]}


def timeline_from_dict(data: dict[str, Any]) -> StepTimeline:
    """Deserialize one timeline."""
    return StepTimeline(int(data["step"]), tuple(Span(**item) for item in data["spans"]))
