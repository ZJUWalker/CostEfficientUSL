"""Render the same timeline contract for measured and simulated runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from usl_ray.profile.timeline import StepTimeline, timeline_from_dict

_OPTIMIZER_OPERATIONS = frozenset({"head_step", "tail_step", "server_step"})


def plot_timelines(
    timelines: tuple[StepTimeline, ...],
    destination: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Write a compact lane-oriented Gantt chart."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    spans = [span for timeline in timelines for span in timeline.spans if span.operation not in _OPTIMIZER_OPERATIONS]
    lanes = sorted({span.lane for span in spans})
    lane_index = {lane: index for index, lane in enumerate(lanes)}
    operations = sorted({span.operation for span in spans})
    colors = {operation: plt.cm.tab20(index % 20) for index, operation in enumerate(operations)}
    figure, axis = plt.subplots(figsize=(14, max(3, len(lanes) * 0.7)))
    if spans:
        origin = min(span.start_ms for span in spans)
        for span in spans:
            color = colors[span.operation]
            axis.barh(
                lane_index[span.lane],
                span.duration_ms,
                left=span.start_ms - origin,
                height=0.65,
                color=color,
                label=span.operation,
            )
            if span.microbatch >= 0:
                red, green, blue, _ = color
                text_color = "white" if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5 else "black"
                axis.text(
                    span.start_ms - origin + span.duration_ms / 2,
                    lane_index[span.lane],
                    str(span.microbatch),
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                    color=text_color,
                    clip_on=True,
                )
    axis.set_yticks(range(len(lanes)), lanes)
    axis.set_xlabel("Time (ms)")
    if title:
        axis.set_title(title)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    if unique:
        axis.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_profile(
    profile_path: str | Path,
    destination: str | Path | None = None,
) -> Path:
    """Render only the final steady global step after warmup."""
    source = Path(profile_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    timelines = tuple(timeline_from_dict(item) for item in payload["timelines"])
    warmup_steps = int(payload["config"]["profile"]["warmup_steps"])
    steady_timelines = timelines[warmup_steps:]
    if not steady_timelines:
        raise ValueError(
            f"Profile has {len(timelines)} timeline(s) and {warmup_steps} warmup step(s); no measured step remains"
        )
    final_step = steady_timelines[-1]
    return plot_timelines(
        (final_step,),
        destination or source.with_name("gantt.png"),
        title=f"Steady global step {final_step.step}",
    )
