"""Live-ish demo: synthesize a univariate stream with injected anomalies and a
regime shift, watch it, log history, and render a dashboard.

Run with::

    uv run --extra dashboard python examples/demo_stream.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from mlwatcher import ConsoleSink, HistoryStore, Watcher


def synth_stream(n: int = 600, seed: int = 7):
    """A noisy sine baseline with point spikes and a mid-stream level shift."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 10.0 + 2.0 * np.sin(t / 25.0)
    noise = rng.normal(0, 0.4, size=n)
    signal = base + noise

    # Sustained regime change: +6 level shift from t=350 onward.
    signal[350:] += 6.0

    # Point anomalies: isolated spikes/dips.
    for idx, mag in [(80, 9.0), (150, -7.0), (210, 8.5), (470, -8.0)]:
        signal[idx] += mag

    return signal


def main() -> None:
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    history_path = out_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)

    signal = synth_stream()

    watcher = Watcher(
        sinks=[ConsoleSink()],
        history=HistoryStore(history_path),
        cooldown=0.0,
    )

    print(f"Streaming {len(signal)} points...\n")
    t0 = time.time()
    n_anom = 0
    with watcher:
        for i, value in enumerate(signal):
            obs = watcher.observe(float(value), timestamp=t0 + i)
            n_anom += obs.is_anomaly

    print(f"\nDone. {n_anom} points flagged. History -> {history_path}")

    # Render the dashboard (requires the 'dashboard' extra).
    try:
        from mlwatcher.dashboard import plot_history

        fig_path = out_dir / "dashboard.png"
        plot_history(history_path, save_path=fig_path)
        print(f"Dashboard -> {fig_path}")
    except ImportError as exc:
        print(f"(skipping plot: {exc})")


if __name__ == "__main__":
    main()
