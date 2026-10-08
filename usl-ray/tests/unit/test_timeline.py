"""Shared profile/simulation metric tests."""

from __future__ import annotations

import pytest

from usl_ray.profile.link import LinkModel, fit_link_model
from usl_ray.profile.timeline import Span, StepTimeline, summarize, trimmed_mean


def test_trimmed_mean_matches_legacy_aggregation() -> None:
    """More than two samples drop exactly one minimum and maximum."""
    assert trimmed_mean([1, 100, 2, 3]) == 2.5
    assert trimmed_mean([1, 3]) == 2
    assert trimmed_mean([]) == 0


def test_summary_uses_all_lanes_and_protects_zero_division() -> None:
    """Metrics derive from spans rather than unmaintained counters."""
    timeline = StepTimeline(
        0,
        (
            Span("client", "head_forward", 0, 0, 0, 10),
            Span("uplink", "act_up", 0, 0, 10, 30, 100),
            Span("server/0", "server_forward", 0, 0, 30, 50),
            Span("downlink", "act_down", 0, 0, 50, 70, 200),
            Span("client", "tail_forward", 0, 0, 70, 100),
        ),
    )
    metrics = summarize([timeline])
    assert metrics.batch_train_time_ms == 100
    assert metrics.client_idle_rate == 60
    assert metrics.server_idle_rate == 80
    assert metrics.uplink_bytes == 100
    assert metrics.downlink_bytes == 200
    assert metrics.operation_mean_ms["head_forward"] == 10
    assert summarize([]).batch_train_time_ms == 0


def test_link_model_uses_tc_si_units() -> None:
    """A 4 MiB payload at 230 Mbit/s takes roughly 145.9 ms plus latency."""
    model = LinkModel(fixed_ms=10, mbps=230)
    assert model.transfer_ms(4 * 1024**2) == pytest.approx(155.89, rel=1e-3)


def test_link_fit_recovers_known_parameters() -> None:
    """Linear fitting recovers a synthetic tc link."""
    expected = LinkModel(7, 200)
    samples = {size: expected.transfer_ms(size) for size in (1024, 1024**2, 4 * 1024**2)}
    actual = fit_link_model(samples)
    assert actual.fixed_ms == pytest.approx(7)
    assert actual.mbps == pytest.approx(200)
