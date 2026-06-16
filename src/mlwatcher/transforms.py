"""Detrending / deseasonalizing transforms applied before detection.

A :class:`Detector` assumes a roughly stationary baseline. On trending or
seasonal data that assumption breaks: a slow daily cycle or upward drift looks
like a never-ending "change" to CUSUM. A transform fixes this by learning the
slow baseline online and feeding the detectors the **residual**
``value - baseline``, which *is* stationary. Spikes and genuine shifts still
produce large residuals, so they remain detectable.

All transforms are online (O(1) or O(period) per step) and stateful.

**Freezing.** Call :meth:`freeze` to stop the model adapting to incoming values
while still emitting predictions (the phase clock keeps advancing and any trend
keeps projecting). This prevents a genuine sustained shift from being quietly
absorbed into the baseline — the residual stays elevated until :meth:`unfreeze`.
The :class:`~mlwatcher.watcher.Watcher` drives this automatically with its
``freeze_on_alert`` option.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Trend:
    """Output of a transform for one value."""

    residual: float      # value - prediction; what the detectors score
    prediction: float    # the learned baseline (level [+ trend] [+ season])
    warmup: bool         # True until the model has enough data to be trusted


class Transform(Protocol):
    def apply(self, value: float) -> Trend: ...
    def reset(self) -> None: ...
    def freeze(self) -> None: ...
    def unfreeze(self) -> None: ...
    def rebaseline(self) -> None: ...


class EWMADetrender:
    """Holt's linear (double-exponential) smoothing for level + slow trend.

    Tracks a smoothed level and trend with EWMA updates and returns the
    residual against the one-step prediction. Use this for drifting /
    trending signals without a fixed seasonal period.

    Parameters
    ----------
    alpha:
        Level smoothing in ``(0, 1]``. Smaller = smoother baseline that
        ignores faster wiggles (and adapts more slowly to real shifts).
    beta:
        Trend smoothing in ``[0, 1]``. ``0`` disables trend tracking (pure
        level EWMA), appropriate when there is no persistent slope.
    warmup:
        Number of values to observe before residuals are trusted.
    """

    def __init__(
        self, alpha: float = 0.05, beta: float = 0.0, warmup: int = 10
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.alpha = alpha
        self.beta = beta
        self.warmup = warmup
        self.frozen = False
        self._level: float | None = None
        self._trend = 0.0
        self._snap = False
        self._n = 0

    def reset(self) -> None:
        self.frozen = False
        self._snap = False
        self._level = None
        self._trend = 0.0
        self._n = 0

    def freeze(self) -> None:
        """Stop adapting the baseline to incoming values (see module docs)."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Resume adapting the baseline."""
        self.frozen = False

    def rebaseline(self) -> None:
        """Snap the baseline onto the next value (accept it as the new normal).

        The next residual is ~0, so re-learning a fresh regime doesn't look
        like a slow change.
        """
        self._snap = True

    def apply(self, value: float) -> Trend:
        value = float(value)
        self._n += 1
        if self._level is None:
            self._level = value
            return Trend(residual=0.0, prediction=value, warmup=True)

        if self._snap:
            self._level = value - self._trend
            self._snap = False

        prediction = self._level + self._trend
        residual = value - prediction

        if self.frozen:
            # Hold the model; project the level forward along the established
            # trend so a sustained shift keeps showing in the residual instead
            # of being absorbed.
            self._level = self._level + self._trend
        else:
            prev_level = self._level
            self._level = self.alpha * value + (1 - self.alpha) * prediction
            if self.beta > 0.0:
                self._trend = (
                    self.beta * (self._level - prev_level)
                    + (1 - self.beta) * self._trend
                )

        return Trend(residual, prediction, warmup=self._n <= self.warmup)


class SeasonalDetrender:
    """Additive Holt-Winters smoothing for data with a repeating cycle.

    Learns a level, optional trend, and a per-phase seasonal profile of known
    ``period`` online, and returns the residual after removing all three.
    Use this when the signal repeats on a fixed cycle (e.g. 24 hourly points
    per day, 7 daily points per week).

    Parameters
    ----------
    period:
        Number of steps in one full cycle.
    alpha, beta, gamma:
        Smoothing for level, trend, and the seasonal component respectively.
        ``beta=0`` disables trend. Smaller values = steadier estimates.
    """

    def __init__(
        self,
        period: int,
        alpha: float = 0.05,
        beta: float = 0.0,
        gamma: float = 0.1,
        warmup: int | None = None,
    ) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        for name, v in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.period = period
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Trust residuals only after two full cycles by default: one to seed
        # the seasonal profile, one for it to settle.
        self.warmup = warmup if warmup is not None else 2 * period
        self.frozen = False
        self._level: float | None = None
        self._trend = 0.0
        self._season: list[float] = [0.0] * period
        self._init_buf: list[float] = []
        self._snap = False
        self._n = 0

    def reset(self) -> None:
        self.frozen = False
        self._snap = False
        self._level = None
        self._trend = 0.0
        self._season = [0.0] * self.period
        self._init_buf = []
        self._n = 0

    def freeze(self) -> None:
        """Stop adapting level/trend/season; keep advancing the phase clock."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Resume adapting the seasonal model."""
        self.frozen = False

    def rebaseline(self) -> None:
        """Snap the level onto the next value (accept it as the new normal),
        keeping the learned seasonal shape so the next residual is ~0."""
        self._snap = True

    def apply(self, value: float) -> Trend:
        value = float(value)
        self._n += 1
        m = self.period

        # Seed from the first full cycle before producing real residuals.
        if self._level is None:
            self._init_buf.append(value)
            if len(self._init_buf) < m:
                return Trend(residual=0.0, prediction=value, warmup=True)
            self._level = sum(self._init_buf) / m
            self._season = [x - self._level for x in self._init_buf]
            return Trend(residual=0.0, prediction=value, warmup=True)

        idx = (self._n - 1) % m
        season = self._season[idx]
        if self._snap:
            self._level = value - self._trend - season
            self._snap = False
        prediction = self._level + self._trend + season
        residual = value - prediction

        if self.frozen:
            # Keep the learned seasonal shape and trend; the phase clock keeps
            # advancing (via _n) so predictions still follow the cycle, but a
            # sustained shift is no longer folded into level/season.
            self._level = self._level + self._trend
        else:
            prev_level = self._level
            self._level = (
                self.alpha * (value - season)
                + (1 - self.alpha) * (prev_level + self._trend)
            )
            if self.beta > 0.0:
                self._trend = (
                    self.beta * (self._level - prev_level)
                    + (1 - self.beta) * self._trend
                )
            self._season[idx] = (
                self.gamma * (value - self._level)
                + (1 - self.gamma) * season
            )

        return Trend(residual, prediction, warmup=self._n <= self.warmup)


__all__ = ["Trend", "Transform", "EWMADetrender", "SeasonalDetrender"]
