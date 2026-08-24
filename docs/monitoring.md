# Monitoring & retraining

[← Back to README](../README.md) · Related: [Pipeline](pipeline.md) · [Decisions](design/decisions.md)

## Why this exists

`serving/app.py` returns a valid-looking `200` and a normal-range
confidence score even when the model's predictions have drifted from what
it was trained on - nothing about the response format changes. Detecting
that requires actively comparing current behavior against training-time
behavior; this module does that comparison rather than waiting for a
visible failure.

## Monitoring vs. observability

Two related but different questions:

| | Monitoring | Observability |
|---|---|---|
| Answers | Is the system healthy? | Why is the system behaving this way? |
| Requires | Knowing what to watch in advance | Rich, structured logs of inputs and outputs |
| Example here | `monitor.py`'s four pass/fail checks | Querying `predictions.db` directly for any question - e.g. every prediction on a specific review, or everything scored below 50% confidence |

Both rest on the same foundation: without a prediction log, there's no
monitoring (nothing to check against a baseline) and no observability
(nothing to query). `monitoring/prediction_log.py` is that foundation.

## What gets logged

Every prediction - real API traffic and simulated traffic alike - is
written to a local SQLite log (`monitoring/predictions.db`,
`monitoring/prediction_log.py`), one row per request: the cleaned text, its
token count and out-of-vocabulary count, the predicted class, confidence,
per-class probabilities, latency, model version, a source tag (`api` vs.
`drift_simulation`), and - only for the simulated set - a hand-assigned
true label. `serving/app.py` calls this on every `/predict` request,
wrapped so a logging failure can never turn a working prediction into a
failed one.

To inspect the log directly:

```bash
python -c "from monitoring import prediction_log; print(prediction_log.read_predictions().to_string())"
```

`read_predictions()` also takes `source=` (`"api"` or `"drift_simulation"`)
and `limit=` to filter or cap what comes back. Or open
`monitoring/predictions.db` directly with any SQLite browser (e.g.
PyCharm's Database tool, or DB Browser for SQLite) - the table is named
`predictions`.

## The baseline

Every comparison below needs something to compare against.
`monitoring/baseline.py` scores the held-out test split (the same split
`rnn_lstm`'s reported macro-F1 is measured on, [Decisions
§7](design/decisions.md)) with the served model and saves: the full array
of confidence scores, the predicted POSITIVE share, and the
out-of-vocabulary rate. Re-run it after any retrain, since a new model
needs a new baseline.

```bash
python -m monitoring.baseline
```

## Simulating drift

Real production traffic doesn't exist yet for this project, so
`monitoring/simulate_drift.py` runs a small, hand-written, hand-labeled set
of 30 reviews (`monitoring/drift_samples.py`) through the model instead -
reviews written in modern internet slang and post-2020 travel topics
(workations, contactless check-in, QR-code menus, digital nomads, EV
charging) that the training data, collected years earlier, has never seen.

Scope note: this is a vocabulary/style shift, not concept drift in the
strict statistical sense - the relationship between text and sentiment
isn't shown to change here, only the surface language is new. Genuine
concept drift would need real traffic collected over time; a synthetic set
can't demonstrate that honestly. What this set does measure directly:
accuracy on text the model has never seen the vocabulary for.

```bash
python -m monitoring.simulate_drift
```

Result on the current model (`rnn_lstm_v1`):

| | Held-out test split (baseline) | Drift set (30 modern-slang reviews) |
|---|---|---|
| Accuracy | 94.34% | 76.67% |
| Macro-F1 | 0.8918 | 0.7600 |
| OOV rate | 0.74% | 5.61% |

A drop this large (macro-F1 down 0.13, accuracy down 17.7 points) on text
that's still clearly, unambiguously positive or negative to a human reader
is exactly the silent-failure symptom this whole system exists to catch -
nothing about the API response itself would tell you this happened.

## Drift types

Four drift types, per the reference taxonomy this project's signals are
mapped against:

| Drift Type | P(X) | P(Y\|X) | P(Y) | Detection | Response |
|---|---|---|---|---|---|
| Covariate Shift | Changes | Stable | Indirect | KS/PSI on input features | Retrain or re-weight on recent data |
| Label Shift | Stable | Stable | Changes | Score distribution shift; label rate monitoring | Recalibrate threshold; retrain on recent data |
| Concept Drift | Stable | Changes | Changes | Ground truth accuracy drop; ADWIN on performance | Retrain on recent data only; old data may be harmful |
| Combined Shift | Changes | Changes | Changes | Multiple signals simultaneously | Full investigation; may require new feature engineering |

## The signal drift report

Before the four trigger signals, `monitoring/monitor.py` prints a plain
descriptive report - not itself a trigger, just a signal-by-signal look at
how the window differs from training, the same way a by-hand comparison
would:

- **Numeric signals** - confidence, token count, and out-of-vocabulary
  count, each shown as a z-score against its baseline distribution
  (`|window mean - baseline mean| / baseline std`), flagged `DRIFTED` past
  1.0 baseline standard deviations, `Stable` otherwise.
- **Categorical signals** - a percentage breakdown of predicted sentiment
  and traffic source in the window.
- **Confidence distribution** - the window bucketed into LOW (<0.60),
  MEDIUM (0.60-0.85), and HIGH (>0.85) confidence bands.

Deliberately called *signals*, not *features*: none of these five are
columns in `feature_store/feature_store.db`. That table does have real
numeric feature columns from the original dataset (`Reviewer_Score`,
`Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`),
but none of them are usable here - a live `/predict` request only ever
sends raw `text`, never the pre-split review halves or score those columns
depend on, so there's no live-request value to compare against a baseline.
Token count and OOV count are used instead because they're the only
numeric quantities computable identically from both the baseline text and
a live request's text - a stand-in for a feature-store comparison, not an
instance of one.

This report exists to make a window's shape legible at a glance before
looking at the pass/fail checks below - useful when a check is borderline
(like vocabulary drift, which sits close to its bar without crossing it)
and a reader wants to see the actual numbers behind that call, not just a
true/false.

## Monitoring signals and retraining triggers

`monitoring/monitor.py` compares a window of logged predictions against the
baseline on four signals, and recommends retraining only when there's real
evidence, not on a fixed schedule:

| Signal | Drift type | What it checks | Threshold | Why this number |
|---|---|---|---|---|
| Confidence drift | General proxy - not one of the four types below | `\|window mean confidence - baseline mean confidence\| / baseline std` | z > 1.0 | The standard per-feature drift cutoff: one baseline standard deviation away from normal |
| Vocabulary drift | Covariate shift - P(X) moving | Out-of-vocabulary rate vs. baseline | +5 percentage points | The training vocabulary is exactly the 20,000 most common training-era words (`train_rnn.py`); a sustained rise is the direct signature of new slang/topics |
| Output drift | Label shift - label rate monitoring, matching the table above exactly | Predicted POSITIVE share vs. baseline (86.5% POSITIVE in the training data, [Decisions §2](design/decisions.md)) | ±10 percentage points | A swing this large means the model's output shape changed, regardless of whether any single feature explains why |
| Performance drop | Concept drift - ground truth accuracy drop, matching the table above exactly | Macro-F1 on the window vs. the model's own reported test-set score (0.8918) | −0.05 or more | Only computable when the window has true labels - real traffic never does at request time (the ground-truth problem below) |

Confidence drift is the one signal that doesn't map onto a single row of the
drift-types table above - it isn't a check on input features (P(X)), the
predicted-label rate (P(Y)), or accuracy (P(Y|X)), just a general "does this
window's confidence look different" early-warning check. It's included as a
companion signal, not a fifth drift type.

When two or more of the other three signals fire together - as they do in
the drift-simulation result below (output drift and performance drop both
fire) - that's this project's real, measured instance of **Combined Shift**:
multiple signals firing simultaneously, per the table above.

Confidence drift and vocabulary/output drift use two different techniques
for the same underlying question - has this distribution moved enough to
matter. Confidence drift is a numeric-signal z-score,
`|window mean - baseline mean| / baseline standard deviation`, flagged past
1.0 baseline standard deviations. Vocabulary and output drift are compared
by raw percentage-point shift instead, the categorical-signal equivalent,
since both are already rates/shares rather than a raw score distribution.

```bash
python -m monitoring.monitor --source drift_simulation
python -m monitoring.monitor --source api
```

On the drift set above, two of the four signals fire (output drift and the
performance drop). Confidence drift stays under its z-score threshold, and
vocabulary drift comes close at +4.87pp but stays just under the 5pp bar -
still, the tool recommends retraining, since a measured accuracy drop and a
16-point swing in the output mix are strong enough evidence on their own.

A window under 30 predictions is treated as too small to trust, since any of
these checks can fire from pure sample noise on a handful of requests
(found this by hand: two live test requests, one of each class, already
swung the predicted-class share by 33 points) - so a small window is
reported as a warning, not a retrain recommendation.

**What doesn't justify retraining, even if it looks alarming in isolation:**

- A single signal firing on a below-`MIN_WINDOW_SIZE` window - noise, not
  evidence, per the point above.
- One check sitting close to its threshold without crossing it (vocabulary
  drift at +4.87pp, just under the +5pp bar, is a real example of this
  project's own data) - close is not the same as fired.
- A retrain decision made on a fixed schedule rather than a measured signal
  - nothing in this project retrains on a calendar; `dvc repro --force
  train_rnn` only ever runs because a check fired.

## The ground-truth problem

Real `/predict` traffic carries no ground-truth label at request time, so
the performance-drop signal is only ever computable on the hand-labeled
drift set, not on live traffic - a separate labeling process would be
needed to change that. Vocabulary drift and output-balance drift don't have
this limitation: both are computable from the input text and the model's
own output alone, which is why they can fire before any labeled example
confirms a problem.

## Worked example: the retraining decision

Reading the four checks isn't the same as deciding whether to retrain. Walked
through step by step, on the drift-simulation run above:

| Question | Answer |
|---|---|
| What signals fired? | Output drift and performance drop (macro-F1 0.8918 → 0.76). Confidence drift (z-score 0.42) and vocabulary drift (+4.87pp) both stayed under their thresholds. |
| Consistent with covariate shift? | Yes - OOV rate rose from 0.74% to 5.61%, close to the alert bar. The drift set is built from modern slang and post-2020 topics the training vocabulary has never seen, which is exactly what covariate shift (input distribution moving) looks like here. |
| Is this concept drift, confirmed? | Yes, for this window specifically - unlike live traffic, the drift set carries hand-assigned true labels, so the macro-F1 drop is a measured fact, not a guess. |
| Decision | Retraining is evidence-justified: the performance drop is directly measured, not inferred, and the output mix moved by 16 points. Confidence and vocabulary staying under threshold doesn't outweigh a measured accuracy drop this large. |

Run the same check against real API traffic (`--source api`) instead of the
drift set, and the picture changes: output drift can still fire, but
performance drop always comes back `null` - there's no ground truth to
score against, and confidence/vocabulary drift need a bigger, more sustained
shift than any handful of test requests will show. That's the harder, more
realistic case: one signal suggesting something moved, no proof yet that the
model is actually wrong, and (until 30+ requests have landed) not even
enough traffic to trust that one. The right move there isn't to retrain on
suspicion - it's to keep collecting logged traffic, and treat the
recommendation as a lead worth investigating, not a queued action.

## What this is not

This is a scoped, honestly-limited implementation, consistent with how the
rest of this project favors dependency-light, locally-runnable tools over
heavier ones ([Decisions §4](design/decisions.md),
[§9](design/decisions.md)). It does not include: a streaming/real-time
detector like ADWIN, an automated pipeline that actually retrains and
redeploys when a trigger fires (retraining still happens by hand, `dvc
repro --force train_rnn`), or fairness/subgroup monitoring. Given evidence
of drift, the response here is a clear, actionable report - not an
autonomous action - which is the appropriate boundary for a project this
size.

Also out of scope: drift in `feature_store.db`'s own real numeric columns
(`Reviewer_Score`, the two `Review_Total_*_Word_Counts` columns) measured
*over time* - comparing today's feature store against an older,
DVC-tracked snapshot of it. Not built because there's only ever been one
snapshot to date (the one-time Kaggle pull); there's nothing yet to
compare it against. Worth adding the moment the raw dataset is re-ingested
and a second snapshot exists.
