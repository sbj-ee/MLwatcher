"""MLwatcher — online anomaly & change detection for univariate streams."""

from .alerts import (
    Alert,
    AlertSink,
    CallbackSink,
    ConsoleSink,
    WebhookSink,
)
from .detectors import CUSUM, Detection, Detector, RobustZScore
from .history import HistoryStore, load_history
from .sources import csv_stream
from .transforms import EWMADetrender, SeasonalDetrender, Transform, Trend
from .watcher import Observation, Watcher, default_detectors

__version__ = "0.1.0"

__all__ = [
    "Watcher",
    "Observation",
    "default_detectors",
    "RobustZScore",
    "CUSUM",
    "Detection",
    "Detector",
    "Alert",
    "AlertSink",
    "ConsoleSink",
    "CallbackSink",
    "WebhookSink",
    "HistoryStore",
    "load_history",
    "csv_stream",
    "EWMADetrender",
    "SeasonalDetrender",
    "Transform",
    "Trend",
]
