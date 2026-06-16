"""Command-line entry point: watch a CSV as a simulated live stream.

Examples
--------
    uv run python -m mlwatcher data.csv --value-column temperature
    uv run python -m mlwatcher data.csv -v 1 -t 0 --no-header --history out.jsonl
    uv run --extra dashboard python -m mlwatcher data.csv -v value --plot dash.png
"""

from __future__ import annotations

import argparse
import sys

from .alerts import AlertSink, ConsoleSink, WebhookSink
from .history import HistoryStore
from .sources import csv_stream
from .transforms import EWMADetrender, SeasonalDetrender, Transform
from .watcher import Watcher, default_detectors


def _coerce_column(spec: str | None) -> str | int | None:
    if spec is None:
        return None
    return int(spec) if spec.lstrip("-").isdigit() else spec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlwatcher",
        description="Watch a univariate CSV stream for anomalies & changes.",
    )
    p.add_argument("csv", help="path to the input CSV")
    p.add_argument(
        "-v", "--value-column", default="0",
        help="metric column: header name or 0-based index (default: 0)",
    )
    p.add_argument(
        "-t", "--time-column", default=None,
        help="optional timestamp column (epoch seconds): name or index",
    )
    p.add_argument(
        "--no-header", action="store_true",
        help="treat the CSV as having no header row",
    )
    p.add_argument("--delimiter", default=",", help="field delimiter")
    p.add_argument(
        "--window", type=int, default=50,
        help="rolling window size for the detectors (default: 50)",
    )
    p.add_argument(
        "--replay-speed", type=float, default=None,
        help="pace rows by their timestamps at this speed multiplier "
             "(needs --time-column; e.g. 10 = 10x real time)",
    )
    p.add_argument(
        "--cooldown", type=float, default=0.0,
        help="min seconds between alerts per detector (default: 0)",
    )
    p.add_argument(
        "--detrend", action="store_true",
        help="EWMA-detrend the signal before scoring (handles slow trend/drift)",
    )
    p.add_argument(
        "--period", type=int, default=None,
        help="enable seasonal (Holt-Winters) detrending with this cycle length "
             "in steps, e.g. 24 for hourly-over-a-day; implies detrending",
    )
    p.add_argument(
        "--ewma-alpha", type=float, default=0.05,
        help="level smoothing for detrending (default: 0.05)",
    )
    p.add_argument(
        "--ewma-beta", type=float, default=0.0,
        help="trend smoothing for detrending; 0 disables trend (default: 0)",
    )
    p.add_argument(
        "--season-gamma", type=float, default=0.1,
        help="seasonal smoothing for --period detrending (default: 0.1)",
    )
    p.add_argument(
        "--freeze-on-alert", action="store_true",
        help="freeze the detrender baseline on the first alert so a sustained "
             "shift stays flagged instead of being absorbed (needs detrending)",
    )
    p.add_argument("--history", default=None, help="write JSONL score history here")
    p.add_argument("--webhook", default=None, help="POST alerts to this URL")
    p.add_argument(
        "--plot", default=None,
        help="render a dashboard PNG to this path after the run "
             "(requires the 'dashboard' extra and --history)",
    )
    p.add_argument("--quiet", action="store_true", help="suppress console alerts")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    sinks: list[AlertSink] = []
    if not args.quiet:
        sinks.append(ConsoleSink())
    if args.webhook:
        sinks.append(WebhookSink(
            args.webhook,
            on_error=lambda e: print(f"webhook error: {e}", file=sys.stderr),
        ))

    detrender: Transform | None = None
    if args.period is not None:
        detrender = SeasonalDetrender(
            period=args.period,
            alpha=args.ewma_alpha,
            beta=args.ewma_beta,
            gamma=args.season_gamma,
        )
    elif args.detrend:
        detrender = EWMADetrender(alpha=args.ewma_alpha, beta=args.ewma_beta)

    # CLI replays a file, so start the history fresh rather than appending.
    history = HistoryStore(args.history, append=False) if args.history else None
    if args.freeze_on_alert and detrender is None:
        print("--freeze-on-alert requires --detrend or --period", file=sys.stderr)
        return 2

    watcher = Watcher(
        detectors=default_detectors(window=args.window),
        sinks=sinks,
        history=history,
        cooldown=args.cooldown,
        detrender=detrender,
        freeze_on_alert=args.freeze_on_alert,
    )

    # --value-column has a default, so this is never None (unlike --time-column).
    value_column = _coerce_column(args.value_column)
    assert value_column is not None
    stream = csv_stream(
        args.csv,
        value_column=value_column,
        time_column=_coerce_column(args.time_column),
        has_header=not args.no_header,
        delimiter=args.delimiter,
        replay_speed=args.replay_speed,
    )

    n = anomalies = 0
    with watcher:
        for ts, value in stream:
            obs = watcher.observe(value, timestamp=ts)
            n += 1
            anomalies += obs.is_anomaly

    print(f"\nProcessed {n} points; {anomalies} flagged.", file=sys.stderr)
    if args.history:
        print(f"History -> {args.history}", file=sys.stderr)

    if args.plot:
        if not args.history:
            print("--plot requires --history", file=sys.stderr)
            return 2
        from .dashboard import plot_history

        plot_history(args.history, save_path=args.plot)
        print(f"Dashboard -> {args.plot}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
