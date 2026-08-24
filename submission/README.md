# Submission artifacts

Four files, each covering one part of the [submission checklist](../README.md#evaluator-quick-links).
Everything here is real output from actually running the code — nothing hand-written.

## `drift_monitor_report.txt`

The full, unedited console output of:

```bash
python -m monitoring.monitor --source drift_simulation
```

This runs the 30-review, hand-labeled modern-slang drift set through the
model, compares it against the training-time baseline on four signals, and
recommends retraining or not. How to read it:

- The first line, **`RETRAIN RECOMMENDED - triggered by: ...`**, is the verdict.
- Each of the four checks (`confidence_drift`, `oov_drift`,
  `output_balance_drift`, `performance_drop`) reports the measured value,
  the baseline value it's compared against, the gap between them, and
  whether that gap crossed the check's threshold (`"triggered": true/false`).
- `retrain_recommended` and `triggered_by` at the bottom summarize which
  checks actually fired. Full explanation of what each check means and why
  its threshold is set where it is: [docs/monitoring.md](../docs/monitoring.md).

## `predictions_log.csv`

An export of every row in `monitoring/predictions.db` — one row per
prediction the API has ever scored, real traffic and the simulated drift
set both included. Column meanings:

| Column | Meaning |
|---|---|
| `id` | Unique row id |
| `timestamp_utc` | When the prediction was made |
| `cleaned_text` | The review text after cleaning (contractions expanded, negations attached, etc.) |
| `token_count` | Number of words in the cleaned text |
| `oov_token_count` | How many of those words aren't in the model's training vocabulary |
| `sentiment` | The model's predicted class - `NEGATIVE` or `POSITIVE` |
| `confidence` | Probability of the winning class (0.0-1.0) |
| `prob_negative` / `prob_positive` | Both class probabilities |
| `latency_ms` | How long that request took to score |
| `model_version` | Which model artifact answered (e.g. `rnn_lstm_v1`) |
| `source` | `api` (a real `/predict` request) or `drift_simulation` (the 30-review synthetic set) |
| `true_sentiment` | The hand-assigned correct answer - only filled in for `drift_simulation` rows, since real API traffic has no ground truth at request time |

To regenerate this file, or filter it (e.g. only real traffic):

```bash
python -c "from monitoring import prediction_log; print(prediction_log.read_predictions(source='api').to_string())"
```

## `api_test_results.txt`

Live output of 11 checks (status codes + response shape) run directly
against the running API - the automated version of the `curl` examples in
[`serving/README.md`](../serving/README.md#endpoints). 11/11 passed.

## `MLflow Experiment Tracking - What Was Done.docx`

A write-up of what's tracked in MLflow for this project - what gets logged
per run, a comparison of the four trained models' real logged parameters
and metrics, and three annotated screenshots of the actual MLflow UI.
