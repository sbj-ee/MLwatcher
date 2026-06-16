"""Input sources for the watcher.

The CSV source lets you *simulate* a live stream from a recorded file: it
yields ``(timestamp, value)`` pairs one row at a time, optionally pacing them
in wall-clock time so the run feels like the real feed.

:func:`tclab_stream` is a *real* live source: it reads a temperature channel
off a connected `TCLab <https://apmonitor.com/heat.htm>`_ Arduino once per
period. Requires the optional ``tclab`` extra (``uv sync --extra tclab``).
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any


def csv_stream(
    path: str | Path,
    value_column: str | int = 0,
    time_column: str | int | None = None,
    has_header: bool = True,
    delimiter: str = ",",
    replay_speed: float | None = None,
) -> Iterator[tuple[float | None, float]]:
    """Yield ``(timestamp, value)`` from a CSV, simulating a live stream.

    Parameters
    ----------
    path:
        CSV file to read.
    value_column:
        Column holding the metric — a header name (if ``has_header``) or a
        0-based index.
    time_column:
        Optional column holding a numeric timestamp (epoch seconds). If
        ``None`` the watcher will assign timestamps itself.
    has_header:
        Whether the first row is a header. Required for name-based columns.
    delimiter:
        Field delimiter.
    replay_speed:
        If set and ``time_column`` is given, sleep between rows to reproduce
        the original cadence at this speed multiplier (``1.0`` = real time,
        ``10.0`` = 10x faster). ``None`` yields as fast as possible.

    Yields
    ------
    (timestamp, value):
        ``timestamp`` is ``None`` when no time column is configured.
    """
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows: Iterator[tuple[str | None, str | None]]
        if has_header:
            reader = csv.DictReader(fh, delimiter=delimiter)
            vcol = _resolve_named(reader.fieldnames, value_column, "value_column")
            tcol = (
                _resolve_named(reader.fieldnames, time_column, "time_column")
                if time_column is not None
                else None
            )
            rows = ((row[vcol], row[tcol] if tcol else None) for row in reader)
        else:
            plain = csv.reader(fh, delimiter=delimiter)
            vi = int(value_column)
            ti = int(time_column) if time_column is not None else None
            rows = (
                (r[vi], r[ti] if ti is not None else None)
                for r in plain
                if r
            )

        prev_ts: float | None = None
        for raw_value, raw_ts in rows:
            if raw_value is None or raw_value == "":
                continue  # skip blank metric cells
            value = float(raw_value)
            ts = float(raw_ts) if raw_ts else None

            if (
                replay_speed
                and replay_speed > 0
                and ts is not None
                and prev_ts is not None
            ):
                wait = (ts - prev_ts) / replay_speed
                if wait > 0:
                    time.sleep(wait)
            prev_ts = ts
            yield ts, value


def tclab_stream(
    channel: str = "T1",
    period: float = 1.0,
    samples: int | None = None,
    *,
    on_tick: Callable[[Any, int], None] | None = None,
    use_model: bool = False,
) -> Iterator[tuple[float, float]]:
    """Stream a live TCLab temperature channel as ``(timestamp, value)``.

    Opens a connection to a TCLab device, then once per ``period`` reads the
    requested channel and yields ``(epoch_seconds, celsius)`` — the shape the
    :class:`~mlwatcher.watcher.Watcher` consumes. The connection is closed
    (heaters off) when the generator is exhausted or the caller stops iterating.

    Parameters
    ----------
    channel:
        Attribute to read each tick, e.g. ``"T1"`` or ``"T2"``.
    period:
        Seconds between samples. TCLab updates at ~1 Hz, so ``1.0`` is natural.
    samples:
        Stop after this many readings. ``None`` streams until the caller stops.
    on_tick:
        Optional ``callback(lab, i)`` run each tick *before* the read, with the
        live device handle and 0-based sample index. Use it to drive the heaters
        (e.g. ``lab.Q1(60)``) so there's a real change to detect.
    use_model:
        Use ``tclab.TCLabModel`` (a digital twin, no hardware) instead of the
        real device. Handy for trying the pipeline without an Arduino attached.

    Yields
    ------
    (timestamp, value):
        Wall-clock epoch seconds and the channel reading in °C.
    """
    try:
        import tclab
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "tclab is required for tclab_stream(). "
            "Install with: uv sync --extra tclab"
        ) from exc

    lab_cls = tclab.TCLabModel if use_model else tclab.TCLab
    with lab_cls() as lab:
        i = 0
        while samples is None or i < samples:
            if on_tick is not None:
                on_tick(lab, i)
            yield time.time(), float(getattr(lab, channel))
            i += 1
            if samples is None or i < samples:
                time.sleep(period)


def _resolve_named(
    fieldnames: Sequence[str] | None, column: str | int, label: str
) -> str:
    """Map a name-or-index column spec to an actual header name."""
    if fieldnames is None:
        raise ValueError(f"{label}: CSV has no header row")
    if isinstance(column, int):
        if column < 0 or column >= len(fieldnames):
            raise ValueError(f"{label}: index {column} out of range")
        return fieldnames[column]
    if column not in fieldnames:
        raise ValueError(
            f"{label}: column {column!r} not in header {list(fieldnames)}"
        )
    return column


__all__ = ["csv_stream", "tclab_stream"]
