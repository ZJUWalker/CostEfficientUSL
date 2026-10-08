"""Render the same timeline contract for measured and simulated runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from usl_ray.profile.timeline import StepTimeline, timeline_from_dict


def plot_timelines(timelines: tuple[StepTimeline, ...], destination: str | Path) -> Path:
    """Write a compact lane-oriented Gantt chart."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    spans = [span for timeline in timelines for span in timeline.spans]
    lanes = sorted({span.lane for span in spans})
    lane_index = {lane: index for index, lane in enumerate(lanes)}
    operations = sorted({span.operation for span in spans})
    colors = {operation: plt.cm.tab20(index % 20) for index, operation in enumerate(operations)}
    figure, axis = plt.subplots(figsize=(14, max(3, len(lanes) * 0.7)))
    if spans:
        origin = min(span.start_ms for span in spans)
        for span in spans:
            axis.barh(
                lane_index[span.lane],
                span.duration_ms,
                left=span.start_ms - origin,
                height=0.65,
                color=colors[span.operation],
                label=span.operation,
            )
    axis.set_yticks(range(len(lanes)), lanes)
    axis.set_xlabel("Time (ms)")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    if unique:
        axis.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_profile(profile_path: str | Path, destination: str | Path | None = None) -> Path:
    """Load profile JSON and render its timelines."""
    source = Path(profile_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    timelines = tuple(timeline_from_dict(item) for item in payload["timelines"])
    return plot_timelines(timelines, destination or source.with_name("gantt.png"))
