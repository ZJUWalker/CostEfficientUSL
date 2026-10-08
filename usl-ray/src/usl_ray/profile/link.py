"""Measured network model shared by runtime checks and simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinkModel:
    """Linear object-transfer model."""

    fixed_ms: float
    mbps: float

    def transfer_ms(self, nbytes: int) -> float:
        """Predict transfer time using tc's SI Mbit/s unit."""
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if self.mbps <= 0:
            return self.fixed_ms
        return self.fixed_ms + nbytes * 8 / (self.mbps * 1_000_000) * 1000


def fit_link_model(samples: dict[int, float]) -> LinkModel:
    """Fit elapsed milliseconds against transferred bytes."""
    if len(samples) < 2:
        raise ValueError("At least two transfer sizes are required")
    sizes = np.asarray(sorted(samples), dtype=np.float64)
    durations = np.asarray([samples[int(size)] for size in sizes], dtype=np.float64)
    slope, intercept = np.polyfit(sizes, durations, 1)
    if slope <= 0:
        raise ValueError(f"Measured transfer slope must be positive, got {slope}")
    return LinkModel(max(0.0, float(intercept)), float(8_000 / slope / 1_000_000))
