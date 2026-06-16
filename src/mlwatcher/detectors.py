"""Online anomaly / change detectors for univariate streams.

Each detector consumes one value at a time and returns a :class:`Detection`.
Detectors are stateful and designed for streaming use: cost per update is
O(window) or O(1), never a full re-fit over history.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Detection:
    """Result of scoring a single value with one detector."""

    detector: str
    value: float
    score: float
    threshold: float
    is_anomaly: bool
    kind: str  # "point" | "change"
    warmup: bool = False
    info: dict = field(default_factory=dict)


class Detector(Protocol):
    """Streaming detector interface."""

    name: str

    def update(self, value: float) -> Detection:
        """Score ``value`` and advance internal state."""

    def reset(self) -> None:
        """Clear all accumulated state."""


# Scale factor that makes MAD a consistent estimator of std for normal data.
_MAD_TO_STD = 1.4826


class RobustZScore:
    """Point-anomaly detector using a rolling median + MAD.

    Flags individual values that sit far from the recent typical level.
    Median/MAD are used instead of mean/std so that a single large spike
    cannot inflate the baseline and mask later anomalies.

    Parameters
    ----------
    window:
        Number of recent values used to estimate the baseline.
    threshold:
        Robust z-score above which a value is flagged.
    min_samples:
        Values seen before the detector starts scoring (warmup).
    """

    def __init__(
        self,
        window: int = 50,
        threshold: float = 4.0,
        min_samples: int | None = None,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.name = "robust_zscore"
        self.window = window
        self.threshold = threshold
        self.min_samples = min_samples or max(10, window // 2)
        self._buf: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    def update(self, value: float) -> Detection:
        value = float(value)
        buf = self._buf
        if len(buf) < self.min_samples:
            # Not enough history yet: record and report warmup.
            buf.append(value)
            return Detection(
                self.name, value, 0.0, self.threshold,
                is_anomaly=False, kind="point", warmup=True,
            )

        ordered = sorted(buf)
        median = _median_sorted(ordered)
        mad = _median_sorted(sorted(abs(x - median) for x in buf))
        scale = _MAD_TO_STD * mad
        if scale <= 1e-12:
            # Degenerate (constant) window: fall back to a tiny epsilon so a
            # genuine jump still scores high instead of dividing by zero.
            scale = 1e-9

        score = abs(value - median) / scale
        is_anom = bool(score > self.threshold)
        # Append after scoring so the current point never biases its own
        # baseline. The median keeps the window robust to the new value.
        buf.append(value)
        return Detection(
            self.name, value, float(score), self.threshold,
            is_anomaly=is_anom, kind="point",
            info={"median": float(median), "mad": float(mad)},
        )


class CUSUM:
    """Two-sided CUSUM change detector for sustained level shifts.

    Standardises each value against a slowly-adapting robust baseline, then
    accumulates positive/negative drift. When the cumulative sum exceeds the
    decision interval ``h`` a change is flagged and the accumulators reset,
    so the detector is ready for the next regime.

    Parameters
    ----------
    window:
        Rolling window used to estimate the baseline mean/scale.
    k:
        Slack (in std units). Drift smaller than ``k`` per step is treated as
        noise; ``0.5`` targets a shift of ~1 std.
    h:
        Decision interval (in std units). Larger = fewer false alarms, slower.
    min_samples:
        Values seen before scoring begins (warmup).
    """

    def __init__(
        self,
        window: int = 50,
        k: float = 0.5,
        h: float = 7.0,
        min_samples: int | None = None,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.name = "cusum"
        self.window = window
        self.k = k
        self.h = h
        self.min_samples = min_samples or max(10, window // 2)
        self._buf: deque[float] = deque(maxlen=window)
        self._s_hi = 0.0
        self._s_lo = 0.0

    def reset(self) -> None:
        self._buf.clear()
        self._s_hi = 0.0
        self._s_lo = 0.0

    def update(self, value: float) -> Detection:
        value = float(value)
        buf = self._buf
        if len(buf) < self.min_samples:
            buf.append(value)
            return Detection(
                self.name, value, 0.0, self.h,
                is_anomaly=False, kind="change", warmup=True,
            )

        ordered = sorted(buf)
        median = _median_sorted(ordered)
        mad = _median_sorted(sorted(abs(x - median) for x in buf))
        scale = _MAD_TO_STD * mad
        if scale <= 1e-12:
            scale = 1e-9

        z = (value - median) / scale
        self._s_hi = max(0.0, self._s_hi + z - self.k)
        self._s_lo = max(0.0, self._s_lo - z - self.k)
        score = max(self._s_hi, self._s_lo)
        is_change = bool(score > self.h)

        direction = 0
        if is_change:
            direction = 1 if self._s_hi >= self._s_lo else -1
            # Reset accumulators and re-baseline on the new regime.
            self._s_hi = 0.0
            self._s_lo = 0.0
            buf.clear()
        buf.append(value)

        return Detection(
            self.name, value, float(score), self.h,
            is_anomaly=is_change, kind="change",
            info={"direction": direction, "z": float(z)},
        )


def _median_sorted(ordered: list[float]) -> float:
    """Median of an already-sorted sequence."""
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


__all__ = ["Detection", "Detector", "RobustZScore", "CUSUM"]
