"""The Watcher: fan one streaming value out to detectors, history, and alerts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .alerts import Alert, AlertSink, ConsoleSink
from .detectors import Detector, Detection, CUSUM, RobustZScore
from .history import HistoryStore
from .transforms import Transform


@dataclass
class Observation:
    """What the watcher returns for each value fed to it."""

    timestamp: float
    value: float
    detections: list[Detection]
    alerts: list[Alert] = field(default_factory=list)
    residual: float | None = None      # value - baseline, if detrended
    prediction: float | None = None    # learned baseline, if detrended
    frozen: bool = False               # detrender baseline frozen after alert

    @property
    def is_anomaly(self) -> bool:
        return any(d.is_anomaly for d in self.detections)


def default_detectors(window: int = 50) -> list[Detector]:
    """A sensible point + change detector pair."""
    return [RobustZScore(window=window), CUSUM(window=window)]


class Watcher:
    """Monitor a univariate stream for anomalies and regime changes.

    Feed values in one at a time with :meth:`observe` (live) or in bulk with
    :meth:`run` (replay / backtest). Each value is scored by every detector;
    threshold crossings become :class:`Alert` objects routed to every sink,
    and — if configured — the full scored point is appended to history.

    Parameters
    ----------
    detectors:
        Detectors to run. Defaults to :func:`default_detectors`.
    sinks:
        Alert destinations. Defaults to a single console sink.
    history:
        Optional :class:`HistoryStore` (or path) to log every scored point.
    cooldown:
        Minimum seconds between alerts from the same detector, to suppress
        bursts. ``0`` disables throttling.
    detrender:
        Optional :class:`~mlwatcher.transforms.Transform` (e.g.
        ``EWMADetrender`` or ``SeasonalDetrender``) applied before scoring.
        The detectors then run on the stationary residual, so trend/season
        doesn't masquerade as anomalies. Alerts are suppressed while the
        transform is still warming up.
    freeze_on_alert:
        When True and a ``detrender`` is set, freeze the detrender's baseline
        the moment an alert fires, so a sustained shift is *not* absorbed and
        keeps showing in the residual. Call :meth:`acknowledge` to thaw and let
        the baseline re-learn the new regime.
    """

    def __init__(
        self,
        detectors: Sequence[Detector] | None = None,
        sinks: Sequence[AlertSink] | None = None,
        history: HistoryStore | str | None = None,
        cooldown: float = 0.0,
        detrender: Transform | None = None,
        freeze_on_alert: bool = False,
    ) -> None:
        self.detectors: list[Detector] = list(
            detectors if detectors is not None else default_detectors()
        )
        self.sinks: list[AlertSink] = list(
            sinks if sinks is not None else [ConsoleSink()]
        )
        if isinstance(history, str) or hasattr(history, "__fspath__"):
            history = HistoryStore(history)  # type: ignore[arg-type]
        self.history: HistoryStore | None = history
        self.cooldown = cooldown
        self.detrender = detrender
        self.freeze_on_alert = freeze_on_alert
        self.frozen = False
        self._last_alert: dict[str, float] = {}

    def observe(
        self, value: float, timestamp: float | None = None
    ) -> Observation:
        """Score a single live value."""
        ts = time.time() if timestamp is None else timestamp

        # Optionally remove trend/season first; detectors see the residual.
        prediction = residual = None
        scored = value
        if self.detrender is not None:
            trend = self.detrender.apply(value)
            prediction, residual = trend.prediction, trend.residual
            scored = residual
            if trend.warmup:
                # Residuals aren't trustworthy yet (and are placeholder zeros
                # during seeding). Don't feed them to the detectors — that
                # would poison their baselines — and don't alert.
                if self.history is not None:
                    self.history.record(ts, value, [], prediction, residual)
                return Observation(
                    ts, value, [], [],
                    residual=residual, prediction=prediction,
                )

        detections = [d.update(scored) for d in self.detectors]

        alerts: list[Alert] = []
        for det in detections:
            if det.is_anomaly and self._allow(det.detector, ts):
                alert = Alert(
                    timestamp=ts,
                    detector=det.detector,
                    kind=det.kind,
                    value=value,
                    score=det.score,
                    threshold=det.threshold,
                    message=_describe(det, detrended=self.detrender is not None),
                )
                alerts.append(alert)
                for sink in self.sinks:
                    sink(alert)

        # Freeze the baseline on the first alert so the shift isn't absorbed.
        if (
            alerts
            and self.freeze_on_alert
            and self.detrender is not None
            and not self.frozen
        ):
            self.detrender.freeze()
            self.frozen = True

        if self.history is not None:
            self.history.record(
                ts, value, detections, prediction, residual, frozen=self.frozen
            )

        return Observation(
            ts, value, detections, alerts,
            residual=residual, prediction=prediction, frozen=self.frozen,
        )

    def acknowledge(self) -> None:
        """Accept the current regime as the new normal and resume adapting.

        Thaws a ``freeze_on_alert`` freeze: snaps the detrender baseline onto
        the next value (so re-absorption isn't seen as a slow change) and
        re-baselines the detectors from the new level. No-op if no detrender.
        """
        if self.detrender is not None:
            self.detrender.unfreeze()
            self.detrender.rebaseline()
        for d in self.detectors:
            d.reset()
        self.frozen = False
        self._last_alert.clear()

    def run(
        self,
        values: Iterable[float],
        timestamps: Iterable[float] | None = None,
    ) -> list[Observation]:
        """Replay a batch/iterable of values (for backtesting)."""
        ts_iter = iter(timestamps) if timestamps is not None else None
        out: list[Observation] = []
        for v in values:
            ts = next(ts_iter) if ts_iter is not None else None
            out.append(self.observe(v, ts))
        return out

    def reset(self) -> None:
        for d in self.detectors:
            d.reset()
        if self.detrender is not None:
            self.detrender.reset()
        self.frozen = False
        self._last_alert.clear()

    def close(self) -> None:
        if self.history is not None:
            self.history.close()

    def _allow(self, detector: str, ts: float) -> bool:
        if self.cooldown <= 0:
            return True
        last = self._last_alert.get(detector)
        if last is not None and ts - last < self.cooldown:
            return False
        self._last_alert[detector] = ts
        return True

    def __enter__(self) -> "Watcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _describe(det: Detection, detrended: bool = False) -> str:
    if det.kind == "change":
        direction = det.info.get("direction", 0)
        word = "upward " if direction > 0 else "downward " if direction < 0 else ""
        suffix = " in residual" if detrended else ""
        return f"sustained {word}level change detected{suffix}"
    if detrended:
        return f"point anomaly: residual {det.score:.1f}x beyond expected baseline"
    median = det.info.get("median")
    if median is not None:
        return f"point anomaly: value deviates sharply from baseline {median:.4g}"
    return "point anomaly"


__all__ = ["Watcher", "Observation", "default_detectors"]
