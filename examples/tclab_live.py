"""Watch a live TCLab Arduino temperature stream with MLwatcher.

Reads T1 once per second and feeds it to a Watcher. Partway through the run it
steps heater Q1 from 0% to a set level, so T1 ramps to a new regime — exactly
the kind of *sustained change* CUSUM is built to catch (and a transient burst of
point anomalies on the way up, collapsed by the cooldown).

Run it (needs the device connected on /dev/ttyACM* and the tclab + dashboard
extras):

    uv run --extra tclab --extra dashboard python examples/tclab_live.py

No Arduino handy? Use the built-in digital twin:

    uv run --extra tclab python examples/tclab_live.py --model
"""

from __future__ import annotations

import argparse
from typing import Any

from mlwatcher import (
    CUSUM,
    ConsoleSink,
    HistoryStore,
    RobustZScore,
    Watcher,
    tclab_stream,
)

STEP_AT = 30      # sample index (≈ seconds) at which to switch the heater on
HEATER = 60       # heater Q1 power, percent
SAMPLES = 180     # total readings ≈ 3 minutes at 1 Hz
HIST_PATH = "examples/out/tclab.jsonl"


def drive_heater(lab: Any, i: int) -> None:
    """Step Q1 on at STEP_AT so there's a real change to detect."""
    lab.Q1(HEATER if i >= STEP_AT else 0.0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", action="store_true",
        help="use the TCLab digital twin instead of real hardware",
    )
    p.add_argument("--samples", type=int, default=SAMPLES)
    args = p.parse_args(argv)

    watcher = Watcher(
        detectors=[RobustZScore(window=30), CUSUM(window=30)],
        sinks=[ConsoleSink()],
        history=HistoryStore(HIST_PATH, append=False),
        cooldown=10.0,
    )

    print(
        f"Streaming T1 at 1 Hz for {args.samples}s; "
        f"heater steps to {HEATER}% at sample {STEP_AT}.\n"
        "Ctrl-C to stop early (heaters are turned off on exit).\n"
    )
    with watcher:
        for ts, value in tclab_stream(
            "T1",
            period=1.0,
            samples=args.samples,
            on_tick=drive_heater,
            use_model=args.model,
        ):
            obs = watcher.observe(value, timestamp=ts)
            flag = "  <-- ANOMALY" if obs.is_anomaly else ""
            print(f"T1={value:5.2f}°C{flag}")

    print(f"\nHistory -> {HIST_PATH}")
    print(
        "Plot it:  uv run --extra dashboard python -c "
        f"\"from mlwatcher.dashboard import plot_history; "
        f"plot_history('{HIST_PATH}', save_path='examples/out/tclab.png')\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
