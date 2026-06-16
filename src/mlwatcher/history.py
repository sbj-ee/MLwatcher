"""Persistent score history.

Every scored point is appended to a JSONL file so the full signal, per-detector
scores, and flags can be replayed or charted later. Designed for append-only
streaming: one line per timestep, flushed immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

from .detectors import Detection


class HistoryStore:
    """Append-only JSONL log of scored points."""

    def __init__(
        self,
        path: str | Path,
        flush_each: bool = True,
        append: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # append=True resumes an existing log (live use); append=False starts
        # fresh (replay/simulation, where re-running shouldn't pile up rows).
        self._fh = self.path.open("a" if append else "w", encoding="utf-8")
        self._flush_each = flush_each

    def record(
        self,
        timestamp: float,
        value: float,
        detections: list[Detection],
        prediction: float | None = None,
        residual: float | None = None,
        frozen: bool = False,
    ) -> None:
        row = {
            "timestamp": timestamp,
            "value": value,
            "scores": {
                d.detector: {
                    "score": round(d.score, 6),
                    "threshold": d.threshold,
                    "is_anomaly": d.is_anomaly,
                    "warmup": d.warmup,
                }
                for d in detections
            },
            "anomaly": any(d.is_anomaly for d in detections),
        }
        if prediction is not None:
            row["prediction"] = round(prediction, 6)
        if residual is not None:
            row["residual"] = round(residual, 6)
        if frozen:
            row["frozen"] = True
        self._fh.write(json.dumps(row) + "\n")
        if self._flush_each:
            self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "HistoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_history(path: str | Path) -> list[dict]:
    """Read a JSONL history file back into a list of rows."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


__all__ = ["HistoryStore", "load_history"]
