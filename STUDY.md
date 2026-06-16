# STUDY.md — Machine Learning concepts behind MLwatcher

A guided tour of the ideas this project is built on. Every concept is tied back
to where it actually shows up in the code, so you can read a term here and then
go see it work in `src/mlwatcher/`. No prior ML background assumed.

**Contents**
1. [The problem: anomaly & change detection](#1-the-problem-anomaly--change-detection)
2. [How learning is framed here](#2-how-learning-is-framed-here)
3. [Statistics you need](#3-statistics-you-need)
4. [Robustness](#4-robustness)
5. [The detectors](#5-the-detectors)
6. [Time series structure: trend, seasonality, noise](#6-time-series-structure-trend-seasonality-noise)
7. [Smoothing & forecasting: EWMA, Holt, Holt-Winters](#7-smoothing--forecasting-ewma-holt-holt-winters)
8. [Detrending = decomposition + residuals](#8-detrending--decomposition--residuals)
9. [Thresholds and the error trade-off](#9-thresholds-and-the-error-trade-off)
10. [Operational concepts](#10-operational-concepts)
11. [Glossary](#11-glossary)
12. [Going further](#12-going-further)

---

## 1. The problem: anomaly & change detection

An **anomaly** is an observation that doesn't fit the pattern the data has shown
so far. Two flavours matter here:

- **Point anomaly** — a single value that's way off (a spike or dip). One bad
  reading. In code: `RobustZScore` in `detectors.py`.
- **Change point / regime shift** — the *typical level itself* moves and stays
  moved. No single point looks crazy, but the baseline is now different. In
  code: `CUSUM` in `detectors.py`.

These need different tools, which is why MLwatcher runs both detectors at once.
A spike is "one point far from normal"; a regime shift is "many points slightly
off in the *same direction*." A spike detector misses slow shifts; a shift
detector is slow on lone spikes.

> **Anomaly vs. outlier vs. novelty:** loosely synonyms. "Outlier" leans
> statistical (far from the bulk of the data), "anomaly" leans operational
> (something you'd want alerted on), "novelty" means a genuinely new pattern not
> seen in training. Here they all mean "flag this."

---

## 2. How learning is framed here

**Supervised vs. unsupervised.** Supervised learning trains on labelled
examples (this email *is* spam, this one *isn't*). Unsupervised learning gets no
labels and must find structure on its own. MLwatcher is **unsupervised**: nobody
tells it which points are anomalies. It learns "normal" from the data stream and
flags departures. That's typical for monitoring — you rarely have a labelled
history of every failure.

**Batch vs. online (streaming).** A *batch* algorithm sees the whole dataset at
once and can make multiple passes. An *online* algorithm sees one item at a time,
updates its state, and never revisits old data. MLwatcher is **online**: every
detector and transform exposes an `update(value)` / `apply(value)` method that
advances internal state in O(window) or O(1) per point. This is what lets it
"watch an input over time" forever without memory growing.

**State / warmup.** An online model carries **state** (the recent window, running
estimates). Before it has seen enough data, its estimates are meaningless — this
is the **warmup** (or burn-in) period. In code, detectors return
`Detection(..., warmup=True)` until `min_samples` points have arrived, and the
`Watcher` won't alert during a transform's warmup.

---

## 3. Statistics you need

Given a window of recent values, we summarise it with a **center** and a
**spread**.

- **Mean (average):** sum / count. Sensitive to outliers — one huge value drags
  it.
- **Median:** the middle value when sorted. Half the data is below, half above.
  Barely moved by a few extreme values.
- **Standard deviation (std):** typical distance of points from the mean. Also
  outlier-sensitive (it squares distances, so big ones dominate).
- **Variance:** std squared.
- **MAD (Median Absolute Deviation):** the median of the absolute distances from
  the median. A spread measure that, like the median, ignores extremes. In
  `detectors.py`: `mad = median(|x - median|)`.

**Standardizing (the z-score).** To ask "how unusual is this value?" we convert
to standard units:

```
z = (value − center) / spread
```

A z-score of 0 means "right at the center"; ±1 means "one typical spread away";
±3 means "far out." This makes a threshold like "flag if |z| > 4" meaningful
regardless of the data's units or scale.

**The normal distribution & the 68–95–99.7 rule.** Many noisy measurements pile
up in a bell curve (the *normal* / Gaussian distribution). For it, ~68% of values
fall within ±1 std, ~95% within ±2, ~99.7% within ±3. So a z-score of 4 is
genuinely rare (~1 in 16,000) for normal data — a reasonable bar for "anomaly."
(Real data has *heavier tails* than the ideal bell curve, which is exactly why
MLwatcher's default threshold is 4.0 rather than 3.0 — see §9.)

---

## 4. Robustness

A statistic is **robust** if a few crazy values don't wreck it. The median and
MAD are robust; the mean and std are not.

This matters enormously for anomaly detection, because of a chicken-and-egg
problem: you estimate "normal" from recent data, *but that data may contain the
very anomalies you're hunting.* If you use mean/std, a single spike inflates the
spread, raising the bar so the *next* spike slips through, and shifts the center
so the baseline is wrong.

MLwatcher uses a **rolling median + MAD** baseline (`RobustZScore`) precisely so
that outliers don't poison the baseline used to detect outliers. The constant
`1.4826` in the code rescales MAD so that, for normal data, it estimates the same
thing the standard deviation would — letting us keep using familiar z-score
thresholds.

---

## 5. The detectors

### 5.1 Robust z-score (point anomalies)

For each new value:
1. Take the recent window (a fixed-size `deque`).
2. Compute its median and MAD.
3. Score `z = |value − median| / (1.4826 · MAD)`.
4. Flag if `z > threshold` (default 4.0).
5. *Then* add the value to the window.

Step 5 ("score before you store") means a point never inflates its own baseline.
The robust window adapts to genuine slow changes but shrugs off lone spikes.

### 5.2 CUSUM (cumulative sum — change detection)

A spike detector won't notice a slow drift, because no single point is extreme.
**CUSUM** accumulates small, consistent deviations so a persistent shift adds up
even when each step looks innocent.

Standardize each value to `z`, then maintain two running sums:

```
S_hi = max(0, S_hi + z − k)      # detects an upward shift
S_lo = max(0, S_lo − z − k)      # detects a downward shift
```

- **`k` (slack / reference value):** a dead-band, in std units. Drift smaller
  than `k` per step is treated as noise and doesn't accumulate. `k = 0.5` targets
  shifts of about 1 std. The `max(0, …)` resets the sum whenever it would go
  negative, so random noise can't slowly wander it upward.
- **`h` (decision interval / threshold):** when either sum exceeds `h`, declare a
  change, then reset the sums to re-baseline for the next regime. Bigger `h` =
  fewer false alarms but slower detection.

CUSUM is a classic from *statistical process control* (SPC) — the discipline of
watching a manufacturing line for when it drifts out of spec. Anomaly monitoring
borrows heavily from it.

### 5.3 Why an ensemble

Running both detectors is a simple **ensemble**: combine models with
complementary strengths. The point detector owns spikes; CUSUM owns sustained
shifts. When a level shift happens, you'll often see the point detector fire a
short *burst* (until its median catches up) while CUSUM cleanly reports one
"sustained change" — visible in the dashboards in the README.

---

## 6. Time series structure: trend, seasonality, noise

A **time series** is values indexed by time. A useful mental model decomposes it
into three parts:

```
value(t) = level/trend(t)  +  seasonality(t)  +  noise(t)
```

- **Level:** the current baseline value.
- **Trend:** slow, sustained movement of the level (drifting up or down).
- **Seasonality:** a pattern that *repeats on a fixed period* — hourly, daily,
  weekly. "Higher every afternoon" is seasonality with a daily period.
- **Noise (residual):** the random leftover after removing the structure above.

**Stationarity.** A series is (roughly) **stationary** if its statistical
properties — mean, spread — don't change over time. The detectors in §5 assume a
stationary baseline. Trend and seasonality *break* that assumption: a daily cycle
makes the mean rise and fall constantly, so CUSUM sees never-ending "change" and
floods you with false alarms. (The README's seasonal example shows 35 false
alerts from exactly this.)

The fix isn't a better detector — it's to *remove* the predictable structure
first and run the detectors on what's left. That's detrending (§8).

---

## 7. Smoothing & forecasting: EWMA, Holt, Holt-Winters

**Moving average.** Average the last *N* points to smooth out noise and estimate
the level. Simple, but treats a point from *N* steps ago as equal to the newest.

**EWMA (Exponentially Weighted Moving Average).** Weight recent points more,
older points exponentially less:

```
level ← α · value + (1 − α) · level
```

- **`α` (alpha, the smoothing factor), 0–1:** how fast it forgets. Large `α`
  reacts quickly but is jumpy; small `α` is smooth but slow. `α = 0.05` ≈
  "remember roughly the last 20 points." EWMA needs O(1) memory — just the
  current `level` — which is perfect for streaming.

**Holt's linear method (double-exponential smoothing).** EWMA tracks level but
lags on a trend. Holt adds a second EWMA for the *slope*:

```
level ← α · value + (1 − α) · (level + trend)
trend ← β · (level − prev_level) + (1 − β) · trend
```

- **`β` (beta):** smoothing for the trend. `β = 0` disables trend tracking
  (pure level EWMA). In code, `EWMADetrender(alpha, beta)`.

**Holt-Winters (triple-exponential smoothing).** Adds a *third* EWMA, one value
per position in the seasonal cycle, to model seasonality:

```
season[phase] ← γ · (value − level) + (1 − γ) · season[phase]
```

- **`γ` (gamma):** smoothing for the seasonal profile.
- **period:** the cycle length you must supply (e.g. 288 five-minute samples =
  one day). In code, `SeasonalDetrender(period, alpha, beta, gamma)`.

This is the **additive** form (components add up); there's also a *multiplicative*
form for when seasonal swings grow with the level. MLwatcher implements the
online additive version.

---

## 8. Detrending = decomposition + residuals

Put §6 and §7 together. A detrender learns the structured part of the signal
online and subtracts it:

```
prediction = level + trend + season(phase)     # what we expected
residual   = value − prediction                # the surprising part
```

The **residual** is approximately stationary even when the raw signal trends and
cycles — so the §5 detectors work on it cleanly. Crucially, *real* anomalies
survive: a spike or a genuine shift still produces a large residual, because the
learned baseline didn't predict it. This is the heart of `transforms.py` and the
README's "Seasonal / trending data" section.

**Predict-then-update.** Online forecasters score a point against the model
*before* learning from it (so a point isn't judged against a baseline it already
influenced), then fold it in. Same idea as "score before you store" in §5.1.

**The absorption trade-off & freezing.** Because the detrender keeps learning, a
*sustained* shift slowly gets absorbed into the baseline — after a while the
model decides "this is the new normal" and the residual decays back toward zero.
Sometimes that's what you want; sometimes you want the shift to *stay* flagged.
That's the `freeze_on_alert` option: on the first alert, stop adapting the
baseline so the residual holds at the shift size. `acknowledge()` later "accepts
the new normal," snapping the baseline to the current level and resuming. This is
a small example of a recurring ML theme: the **stability–plasticity dilemma** —
adapt too fast and you forget/absorb real signals; adapt too slow and you can't
track genuine change.

---

## 9. Thresholds and the error trade-off

Detection turns a continuous **score** into a yes/no decision via a
**threshold**. Where you put it trades off two error types:

- **False positive (Type I error):** a false alarm — flagged but nothing was
  wrong. Too many and operators ignore the system ("alert fatigue").
- **False negative (Type II error):** a miss — a real anomaly went unflagged.

Lowering the threshold catches more real anomalies (fewer misses) but raises
false alarms, and vice-versa. You can't minimise both at once; you pick a balance
for your cost of a miss vs. cost of a false alarm.

Related vocabulary you'll meet:

- **Precision:** of the points you flagged, what fraction were real?
- **Recall (sensitivity / true-positive rate):** of the real anomalies, what
  fraction did you catch?
- **ROC / precision-recall curves:** plots of these trade-offs as the threshold
  sweeps.
- **ARL (Average Run Length):** an SPC metric for change detectors — the average
  number of points between false alarms on stationary data (ARL₀, want it
  *large*) vs. the average delay to detect a real shift (ARL₁, want it *small*).
  Tuning CUSUM's `h` is exactly trading ARL₀ against ARL₁.

How MLwatcher's defaults were set is a concrete example: the thresholds weren't
guessed, they were chosen by *measuring* the false-alarm rate on synthetic
stationary data and the detection delay on injected shifts, then picking the knee
of the trade-off (`threshold = 4.0`, `h = 7.0`). That empirical tuning loop —
simulate, measure, adjust — is how detection systems are calibrated in practice.

---

## 10. Operational concepts

These aren't "ML theory" but they're what makes a detector usable in production:

- **Warmup / burn-in:** ignore output until the model has enough data (§2).
- **Cooldown / debouncing:** after firing, suppress repeat alerts for a short
  time so one event doesn't spam you. In code: `Watcher(cooldown=…)`.
- **Hysteresis / latching:** require a stronger condition to *leave* a state than
  to enter it, to avoid flapping. `freeze_on_alert` latches into a frozen state
  until you explicitly `acknowledge()`.
- **Drift:** the slow change of "normal" over time (concept drift). Online models
  handle mild drift by continuously adapting; the detrender handles structured
  drift explicitly.
- **Backtesting / replay:** feed historical data through the live code path to
  evaluate it. `Watcher.run(values)` and the CSV source exist for this.

---

## 11. Glossary

| Term | One-line meaning |
|------|------------------|
| Anomaly / outlier | A value that doesn't fit the established pattern. |
| Point anomaly | A single out-of-range value (spike/dip). |
| Change point / regime shift | The baseline level moves and stays moved. |
| Supervised / unsupervised | Learning with / without labelled examples. (Here: unsupervised.) |
| Online / streaming | Process one item at a time, never revisit. (Here: online.) |
| Batch | See all data at once, multiple passes. |
| Warmup / burn-in | Initial period before estimates are trustworthy. |
| Mean / median | Average / middle value (center of the data). |
| Std / variance | Spread of the data (variance = std²). |
| MAD | Median absolute deviation — a robust spread. |
| Robust | Insensitive to a few extreme values. |
| z-score / standardize | (value − center) / spread; "how many spreads out." |
| Normal (Gaussian) distribution | The bell curve; basis of the 68-95-99.7 rule. |
| Heavy tails | Extreme values more common than the bell curve predicts. |
| CUSUM | Cumulative-sum change detector. |
| Slack `k` / interval `h` | CUSUM's noise dead-band / decision threshold. |
| SPC | Statistical process control (industrial monitoring roots). |
| Ensemble | Combining multiple models for complementary strengths. |
| Time series | Values indexed by time. |
| Trend / seasonality / noise | Slow drift / repeating cycle / random residual. |
| Stationary | Statistical properties don't change over time. |
| EWMA | Exponentially weighted moving average (smoothing). |
| α, β, γ | Smoothing rates for level, trend, season. |
| Holt / Holt-Winters | Double / triple exponential smoothing (trend / + season). |
| Detrending / decomposition | Removing structure to expose residuals. |
| Residual | What's left after subtracting the model's prediction. |
| Predict-then-update | Score a point before learning from it. |
| Threshold | Score cutoff that turns a number into a decision. |
| False positive / negative | False alarm / missed detection. |
| Precision / recall | Fraction of flags that were real / of real ones caught. |
| ARL | Avg run length: points between false alarms vs. detection delay. |
| Cooldown / debounce | Suppress repeat alerts for a short window. |
| Hysteresis / latching | Harder to leave a state than to enter it. |
| Concept drift | "Normal" slowly changing over time. |
| Backtesting | Replaying history through the live code to evaluate it. |

---

## 12. Going further

MLwatcher deliberately uses transparent, classical statistics — every decision
is explainable, which matters for monitoring. If you want to go deeper, these are
the natural next topics and how they'd extend this project:

- **Other detectors:** EWMA control charts, Bayesian online change-point
  detection, Isolation Forest, seasonal-hybrid ESD (Twitter's approach),
  Matrix Profile. Each plugs in via the `update(value) -> Detection` interface.
- **Multivariate detection:** watching several correlated signals together
  (Mahalanobis distance, PCA residuals) instead of one channel at a time.
- **Forecasting models:** ARIMA, Prophet, or small neural nets (LSTM/temporal
  CNN) as the baseline predictor, with anomalies = large forecast residuals —
  the same detrend-then-detect pattern as §8, just a fancier predictor.
- **Evaluation:** building a labelled test set and computing precision/recall to
  tune thresholds objectively (§9), instead of by eye.

The throughline: **model what's normal, then alert on the surprise.** Everything
above is variations on that one idea.
