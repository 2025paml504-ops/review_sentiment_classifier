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

## Monitoring signals and retraining triggers

`monitoring/monitor.py` compares a window of logged predictions against the
baseline on four signals, and recommends retraining only when there's real
evidence, not on a fixed schedule:

| Signal | Concept | What it checks | Threshold | Why this number |
|---|---|---|---|---|
| Confidence drift | Proxy signal (z-score) | `\|window mean confidence - baseline mean confidence\| / baseline std` | z > 1.0 | The standard per-feature drift cutoff: one baseline standard deviation away from normal |
| Vocabulary drift | Covariate shift proxy | Out-of-vocabulary rate vs. baseline | +5 percentage points | The training vocabulary is exactly the 20,000 most common training-era words (`train_rnn.py`); a sustained rise is the direct signature of new slang/topics |
| Output drift | Label shift proxy | Predicted POSITIVE share vs. baseline (86.5% POSITIVE in the training data, [Decisions §2](design/decisions.md)) | ±10 percentage points | A swing this large means the model's output shape changed, regardless of whether any single feature explains why |
| Performance drop | Concept drift, limited by ground-truth delay | Macro-F1 on the window vs. the model's own reported test-set score (0.8918) | −0.05 or more | Only computable when the window has true labels - real traffic never does at request time (the ground-truth problem below) |

"Proxy signal" here means the same thing it means in monitoring generally: confidence and predicted-label share are measurable stand-ins for shifts we can't check directly at request time, since real traffic carries no ground truth. Vocabulary and output drift are the closest things this project has to a true covariate-shift / label-shift test - neither is a formal statistical test on the input features themselves, just a direct proxy for one. Performance drop is the only signal that touches concept drift (a change in how features relate to sentiment), and it's gated by ground-truth delay: it only runs on the hand-labeled drift set, never on live traffic.

Confidence drift and vocabulary/output drift use two different techniques
for the same underlying question - has this distribution moved enough to
matter. Confidence drift is a per-feature z-score,
`|window mean - baseline mean| / baseline standard deviation`, flagged past
1.0 baseline standard deviations - the numeric-feature drift check.
Vocabulary and output drift are compared by raw percentage-point shift
instead, the categorical-feature equivalent, since both are already rates/
shares rather than a raw score distribution.

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
