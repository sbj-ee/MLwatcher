"""Alert sinks: where threshold crossings get sent."""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol


@dataclass
class Alert:
    """A single anomaly/change event worth surfacing."""

    timestamp: float
    detector: str
    kind: str  # "point" | "change"
    value: float
    score: float
    threshold: float
    message: str

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "iso_time": datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat(),
            "detector": self.detector,
            "kind": self.kind,
            "value": self.value,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "message": self.message,
        }


class AlertSink(Protocol):
    def __call__(self, alert: Alert) -> None: ...


class ConsoleSink:
    """Print alerts to a stream (stderr by default)."""

    def __init__(self, stream=sys.stderr) -> None:
        self._stream = stream

    def __call__(self, alert: Alert) -> None:
        d = alert.as_dict()
        print(
            f"[{d['iso_time']}] ALERT {d['detector']}/{d['kind']} "
            f"value={d['value']:.4g} score={d['score']:.2f} "
            f"(>{d['threshold']:.2f}) :: {d['message']}",
            file=self._stream,
            flush=True,
        )


class CallbackSink:
    """Adapt any plain callable into a sink."""

    def __init__(self, fn: Callable[[Alert], None]) -> None:
        self._fn = fn

    def __call__(self, alert: Alert) -> None:
        self._fn(alert)


class WebhookSink:
    """POST the alert as JSON to a URL (e.g. Slack incoming webhook).

    Network failures are swallowed and reported to ``on_error`` so a flaky
    endpoint never crashes the monitoring loop.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self._on_error = on_error

    def __call__(self, alert: Alert) -> None:
        payload = json.dumps(alert.as_dict()).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=self.timeout).close()
        except Exception as exc:  # noqa: BLE001 - never break the stream
            if self._on_error is not None:
                self._on_error(exc)


__all__ = ["Alert", "AlertSink", "ConsoleSink", "CallbackSink", "WebhookSink"]
