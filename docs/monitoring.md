# Monitoring & retraining

[← Back to README](../README.md) · Related: [Pipeline](pipeline.md) · [Decisions](design/decisions.md)

## Why this exists

`serving/app.py` returns a valid-looking `200` and a normal-range
confidence score even when the model's predictions have drifted from what
it was trained on - nothing about the response format changes. Detecting
that requires actively comparing current behavior against training-time
behavior; this module does that comparison rather than waiting for a
visible failure.

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

## Monitoring signals and retraining triggers

`monitoring/monitor.py` compares a window of logged predictions against the
baseline on four signals, and recommends retraining only when there's real
evidence, not on a fixed schedule:

| Signal | What it checks | Threshold | Why this number |
|---|---|---|---|
| Confidence drift | Kolmogorov-Smirnov test comparing the window's confidence-score distribution to the baseline's | p < 0.05 | The standard statistical-significance cutoff, not tuned to this data |
| Vocabulary drift | Out-of-vocabulary rate vs. baseline | +5 percentage points | The training vocabulary is exactly the 20,000 most common training-era words (`train_rnn.py`); a sustained rise is the direct signature of new slang/topics |
| Output drift | Predicted POSITIVE share vs. baseline (86.5% POSITIVE in the training data, [Decisions §2](design/decisions.md)) | ±10 percentage points | A swing this large means the model's output shape changed, regardless of whether any single feature explains why |
| Performance drop | Macro-F1 on the window vs. the model's own reported test-set score (0.8918) | −0.05 or more | Only computable when the window has true labels - real traffic never does at request time (the ground-truth problem below) |

```bash
python -m monitoring.monitor --source drift_simulation
python -m monitoring.monitor --source api
```

On the drift set above, three of the four signals fire (confidence drift,
output drift, and the performance drop - vocabulary drift comes close at
+4.87pp but stays just under the 5pp bar), and the tool recommends
retraining. A window under 30 predictions is treated as too small to trust
- any of these checks can fire from pure sample noise on a handful of
requests (found this by hand: two live test requests, one of each class,
already swung the predicted-class share by 33 points) - so a small window
is reported as a warning, not a retrain recommendation.

## The ground-truth problem

Real `/predict` traffic carries no ground-truth label at request time, so
the performance-drop signal is only ever computable on the hand-labeled
drift set, not on live traffic - a separate labeling process would be
needed to change that. Vocabulary drift and output-balance drift don't have
this limitation: both are computable from the input text and the model's
own output alone, which is why they can fire before any labeled example
confirms a problem.

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
