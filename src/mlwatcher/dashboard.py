"""Visualize a stream, its detector scores, and flagged events.

matplotlib is an optional dependency. Install with::

    uv sync --extra dashboard
"""

from __future__ import annotations

from pathlib import Path

from .history import load_history


def plot_history(
    source,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot the signal + per-detector scores from a history file or rows.

    Parameters
    ----------
    source:
        Path to a JSONL history file, or a list of history rows.
    save_path:
        If given, write the figure here (PNG inferred from extension).
    show:
        If True, open an interactive window.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for the dashboard. "
            "Install with: uv sync --extra dashboard"
        ) from exc

    rows = load_history(source) if isinstance(source, (str, Path)) else list(source)
    if not rows:
        raise ValueError("no history rows to plot")

    t = [r["timestamp"] for r in rows]
    t0 = t[0]
    t = [x - t0 for x in t]  # seconds since start, easier to read
    values = [r["value"] for r in rows]

    detector_names = sorted({name for r in rows for name in r["scores"]})
    has_detrend = any("prediction" in r for r in rows)

    n_panels = 1 + (1 if has_detrend else 0) + len(detector_names)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(11, 2.4 * n_panels),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    # Top panel: the raw signal with anomalies marked.
    ax = axes[0]
    ax.plot(t, values, lw=1.0, color="#3366cc", label="signal")
    if has_detrend:
        pred = [r.get("prediction") for r in rows]
        ax.plot(t, pred, lw=1.2, color="#ff9900", ls="--",
                label="learned baseline")
    anom_t = [ti for ti, r in zip(t, rows) if r["anomaly"]]
    anom_v = [r["value"] for r in rows if r["anomaly"]]
    if anom_t:
        ax.scatter(anom_t, anom_v, color="#cc2222", s=28, zorder=5,
                   label="anomaly")
    # Shade spans where the baseline was frozen (sustained-shift hold).
    _shade_spans(
        ax, t, [bool(r.get("frozen")) for r in rows],
        color="#9966cc", alpha=0.12, label="baseline frozen",
    )
    ax.set_ylabel("value")
    ax.set_title("MLwatcher — signal & detector scores")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    panel = 1
    # Optional residual panel: what the detectors actually score post-detrend.
    if has_detrend:
        ax = axes[panel]
        panel += 1
        resid = [r.get("residual", 0.0) for r in rows]
        ax.plot(t, resid, lw=1.0, color="#117733", label="residual (detrended)")
        ax.axhline(0.0, color="#888888", lw=0.8)
        ax.set_ylabel("residual")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

    # One panel per detector: score vs threshold.
    for ax, name in zip(axes[panel:], detector_names):
        scores = [r["scores"].get(name, {}).get("score", 0.0) for r in rows]
        thresh = next(
            (r["scores"][name]["threshold"] for r in rows if name in r["scores"]),
            None,
        )
        ax.plot(t, scores, lw=1.0, color="#444444", label=f"{name} score")
        if thresh is not None:
            ax.axhline(thresh, color="#cc2222", ls="--", lw=1.0,
                       label=f"threshold={thresh:g}")
        flagged_t = [
            ti for ti, r in zip(t, rows)
            if r["scores"].get(name, {}).get("is_anomaly")
        ]
        for ft in flagged_t:
            ax.axvline(ft, color="#cc2222", alpha=0.15)
        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("time (s since start)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    return fig


def _shade_spans(ax, t, mask, color, alpha, label=None):
    """Shade contiguous runs where ``mask`` is True; label only the first."""
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = t[i]
        elif not on and start is not None:
            ax.axvspan(start, t[i], color=color, alpha=alpha, label=label)
            label = None
            start = None
    if start is not None:
        ax.axvspan(start, t[-1], color=color, alpha=alpha, label=label)


__all__ = ["plot_history"]
