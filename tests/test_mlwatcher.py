import numpy as np
import pytest

from mlwatcher import (
    CUSUM,
    EWMADetrender,
    RobustZScore,
    SeasonalDetrender,
    Watcher,
    load_history,
)
from mlwatcher.history import HistoryStore


def _clean(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return 5.0 + rng.normal(0, 0.3, size=n)


def test_robust_zscore_flags_spike_not_noise():
    det = RobustZScore(window=50)
    flagged = []
    for i, v in enumerate(_clean()):
        v = float(v)
        if i == 200:
            v += 10.0  # injected spike
        d = det.update(v)
        if d.is_anomaly:
            flagged.append(i)
    assert 200 in flagged
    # Robust to noise: at most a stray false positive, not a storm.
    assert len(flagged) <= 2


def test_robust_zscore_warmup_never_flags():
    det = RobustZScore(window=50, min_samples=20)
    for v in _clean(n=10):
        d = det.update(float(v))
        assert d.warmup and not d.is_anomaly


def test_cusum_detects_level_shift():
    det = CUSUM(window=50, k=0.5, h=5.0)
    data = list(_clean(n=400))
    for i in range(200, 400):
        data[i] += 3.0  # sustained shift
    changes = [i for i, v in enumerate(data) if det.update(float(v)).is_anomaly]
    assert changes, "expected at least one change point"
    # First detection should land shortly after the shift at 200.
    assert 200 <= changes[0] <= 230


def test_cusum_quiet_on_stationary():
    det = CUSUM(window=50)
    changes = [i for i, v in enumerate(_clean(n=500)) if det.update(float(v)).is_anomaly]
    # Stationary noise: default h gives a high ARL, so at most a rare blip.
    assert len(changes) <= 1


def test_constant_window_no_div_by_zero():
    det = RobustZScore(window=20)
    for _ in range(30):
        det.update(4.0)
    d = det.update(4.0)
    assert not d.is_anomaly
    jump = det.update(50.0)
    assert jump.is_anomaly  # genuine jump still caught


def test_watcher_history_and_alerts(tmp_path):
    alerts = []
    hist_path = tmp_path / "h.jsonl"
    w = Watcher(
        sinks=[lambda a: alerts.append(a)],
        history=HistoryStore(hist_path),
    )
    data = list(_clean(n=200))
    data[150] += 12.0
    w.run(data, timestamps=list(range(len(data))))
    w.close()

    rows = load_history(hist_path)
    assert len(rows) == len(data)
    assert any(r["anomaly"] for r in rows)
    assert any(a.value > 12 for a in alerts)


def test_watcher_cooldown_throttles(tmp_path):
    alerts = []
    w = Watcher(
        detectors=[RobustZScore(window=30)],
        sinks=[lambda a: alerts.append(a)],
        cooldown=5.0,
    )
    # A run of consecutive anomalies at timestamps 40..49.
    base = list(_clean(n=60))
    for i in range(40, 50):
        base[i] += 15.0
    w.run(base, timestamps=[float(i) for i in range(len(base))])
    # Throttled: no two alerts from the same detector within the cooldown.
    times = sorted(a.timestamp for a in alerts)
    assert all(b - a >= 5.0 for a, b in zip(times, times[1:]))
    # And the burst is collapsed to far fewer than 10 alerts.
    assert len(alerts) < 5


def _seasonal(n=800, period=200, amp=5.0, noise=0.3, seed=2):
    # Slow cycle relative to the detector window: the realistic seasonal case
    # (e.g. a daily pattern sampled often) where naive detection floods.
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 20.0 + amp * np.sin(2 * np.pi * t / period) + rng.normal(0, noise, n)


def test_seasonal_data_floods_without_detrending():
    # Strong seasonality looks like constant change to a stationary detector.
    w = Watcher(sinks=[], history=None)
    obs = w.run([float(v) for v in _seasonal()])
    assert sum(o.is_anomaly for o in obs) > 10  # many false alarms


def test_seasonal_detrender_suppresses_false_alarms():
    period = 200
    w = Watcher(
        sinks=[],
        history=None,
        detrender=SeasonalDetrender(period=period, gamma=0.2),
    )
    obs = w.run([float(v) for v in _seasonal(period=period)])
    # After the warmup cycles, the deseasonalized stream should be quiet.
    post_warmup = obs[2 * period:]
    flags = sum(o.is_anomaly for o in post_warmup)
    assert flags <= 2


def test_detrender_still_catches_injected_anomaly():
    period = 200
    data = list(_seasonal(n=1000, period=period))
    data[700] += 15.0  # point spike on top of the seasonal pattern
    for i in range(800, len(data)):
        data[i] += 8.0  # sustained shift on top of the seasonal pattern
    alerts = []
    w = Watcher(
        sinks=[lambda a: alerts.append(a)],
        detrender=SeasonalDetrender(period=period, gamma=0.2),
    )
    w.run([float(v) for v in data], timestamps=list(range(len(data))))
    # The point spike near index 700 must surface.
    assert any(a.detector == "robust_zscore" and abs(a.timestamp - 700) <= 2
               for a in alerts), "missed the point spike"
    # The sustained shift after 800 must register a change.
    assert any(a.detector == "cusum" and a.timestamp >= 800 for a in alerts), \
        "missed the regime shift"


def test_seasonal_detrender_warmup_is_silent():
    period = 24
    det = SeasonalDetrender(period=period)
    for v in _seasonal(n=period, period=period):
        tr = det.apply(float(v))
        assert tr.warmup and tr.residual == 0.0


def test_ewma_detrender_removes_linear_trend():
    # Ramp + noise: residuals should hover near zero once warmed up.
    rng = np.random.default_rng(0)
    n = 400
    data = 0.05 * np.arange(n) + rng.normal(0, 0.2, n)
    det = EWMADetrender(alpha=0.1, beta=0.1)
    resid = [det.apply(float(v)).residual for v in data]
    tail = resid[100:]
    assert abs(np.mean(tail)) < 0.3
    assert np.std(tail) < 1.0


def _shifted_seasonal(n=1000, period=200, shift_at=500, shift=10.0):
    data = list(_seasonal(n=n, period=period))
    for i in range(shift_at, n):
        data[i] += shift
    return data


def test_freeze_on_alert_keeps_sustained_shift_visible():
    period = 200
    data = _shifted_seasonal(period=period)

    # Without freeze, the baseline absorbs the shift and the residual decays.
    w_absorb = Watcher(
        sinks=[], detrender=SeasonalDetrender(period=period, gamma=0.2)
    )
    obs_absorb = w_absorb.run([float(v) for v in data])
    late_absorb = np.mean([abs(o.residual) for o in obs_absorb[-100:]])
    assert late_absorb < 2.0
    assert not w_absorb.frozen

    # With freeze, the baseline is held so the +10 shift stays in the residual.
    w_freeze = Watcher(
        sinks=[],
        detrender=SeasonalDetrender(period=period, gamma=0.2),
        freeze_on_alert=True,
    )
    obs_freeze = w_freeze.run([float(v) for v in data])
    late_freeze = np.mean([abs(o.residual) for o in obs_freeze[-100:]])
    assert w_freeze.frozen
    assert late_freeze > 7.0
    assert all(o.frozen for o in obs_freeze[-50:])


def test_acknowledge_accepts_new_normal():
    period = 200
    data = _shifted_seasonal(n=1200, period=period)
    w = Watcher(
        sinks=[],
        detrender=SeasonalDetrender(period=period, gamma=0.2),
        freeze_on_alert=True,
    )
    w.run([float(v) for v in data[:680]])
    assert w.frozen

    w.acknowledge()
    assert not w.frozen
    obs = w.run([float(v) for v in data[680:]])
    # After accepting the new normal, residuals settle near zero and it does
    # not immediately re-freeze on the re-absorption transient.
    assert not w.frozen
    assert np.mean([abs(o.residual) for o in obs[-100:]]) < 2.0


def test_freeze_on_alert_without_detrender_is_noop():
    data = list(_clean(n=200))
    data[150] += 12.0
    w = Watcher(sinks=[], freeze_on_alert=True)  # no detrender configured
    obs = w.run(data, timestamps=list(range(len(data))))
    assert any(o.is_anomaly for o in obs)
    assert not w.frozen  # nothing to freeze without a detrender


def test_run_matches_observe():
    w1 = Watcher(detectors=[RobustZScore()], sinks=[])
    w2 = Watcher(detectors=[RobustZScore()], sinks=[])
    data = [float(v) for v in _clean(n=100)]
    obs_run = w1.run(data, timestamps=list(range(len(data))))
    obs_one = [w2.observe(v, timestamp=i) for i, v in enumerate(data)]
    assert [o.is_anomaly for o in obs_run] == [o.is_anomaly for o in obs_one]
