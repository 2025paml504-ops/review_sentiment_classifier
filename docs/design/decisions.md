# Decision-making guide

[← Design docs](README.md) · Related: [Architecture](architecture.md) · [Pipeline](../pipeline.md) · [Versioning](../versioning.md)

Lightweight ADR-style records of the key choices behind the pipeline: what
was decided, and why.

## At a glance

| # | Decision | Why |
|---|---|---|
| [1](#1-text-cleaning--tokenization) | Clean text with plain regex, de-duplicate first, merge review fields into one document | Keeps negation ("not good") visible as a feature; no extra dependencies |
| [2](#2-sentiment-labeling--vader-binary) | Label sentiment with VADER on the review text, binary NEGATIVE/POSITIVE | Text-derived labels match what the review actually says |
| [3](#3-tf-idf-fit-on-train-only) | TF-IDF features, fit on the train split only, after a stratified split | Fast, cheap, interpretable; no test-vocabulary leakage |
| [4](#4-feature-storage--sqlite--csvjson) | SQLite feature store; interim data as CSV with tokens as JSON | No server needed; no Parquet dependency |
| [5](#5-schema-contract-as-json) | Expected columns/dtypes live in a JSON file, not code | A schema change becomes a one-line edit |
| [6](#6-versioning-with-dvc) | Data/artifacts versioned with DVC; git holds only pointers | Too large for git; a commit hash still pins an exact version |
| [7](#7-four-models-trained-rnn_lstm-served) | Four models trained (logreg, linear_svc, RNN/LSTM, BERT-tiny); `rnn_lstm` served | Wins on macro-F1 by a clear margin (0.8918 vs. next-best 0.8652) |
| [8](#8-macro-f1-headline-metric--class-weighting) | Macro-F1 is the headline metric; all four models weight the rare class | Accuracy hides poor performance on the small class; weighting measurably helps it |
| [9](#9-mlflow-experiment-tracking) | Every run logs settings, scores, environment, and a hypothesis/conclusion pair | Makes any past run reproducible and comparable from its own record |
| [10](#10-tuning-alternatives-considered-and-not-adopted) | A record of every measured alternative to the current config that wasn't adopted | Stops the same dead end from being tried again by accident |
| [11](#11-plain-html-ui-no-framework) | `ui/index.html` is one plain HTML/CSS/JS file, no framework | One form, one API call - too simple to need framework overhead |
| [12](#12-monitoring-drift-simulation-and-retraining-triggers) | Every prediction is logged; drift is simulated; four signals decide when to recommend retraining | Drift set shows a real, measured macro-F1 drop (0.8918 → 0.7600) |
| [13](#13-api-failure-handling-and-contract-change-policy) | Missing model fails soft (`503`, not a crash); new fields ship optional before required | A caller can tell "broken" from "not ready"; no existing caller can be broken |

---

## 1. Text cleaning & tokenization

**Decision.** `features/build_features.py`: merge `Positive_Review` +
`Negative_Review` into one `full_review` field; de-duplicate on
`[full_review, Reviewer_Score]` first; then lowercase, expand contractions,
strip placeholder text, and attach negators to the next word
(`not_good`). `validation/diagnose_cleaning.py` reports known defect
counts on demand, kept off the DVC DAG.

**Why.** Dependency-light (stdlib `re` only) and keeps negation visible as
a real feature instead of losing it to punctuation stripping.
De-duplicating first avoids wasted cleaning passes on rows that get
dropped anyway.

## 2. Sentiment labeling — VADER, binary

**Decision.** Sentiment comes from VADER's compound score on `full_review`
(`compound >= 0.0` → POSITIVE, else NEGATIVE, `features/build_features.py`),
not from `Reviewer_Score`. 86.5% / 13.5% POSITIVE/NEGATIVE split.

**Why.** Text-derived labels describe what the review actually says; a
numeric-score threshold can disagree with the text (an 8.8-scored review
reading "staff rude unhelpful money grabbers"). VADER's `0.0` cutoff is its
own built-in zero-point, not tuned to hit a target balance.

## 3. TF-IDF, fit on train only

**Decision.** `TfidfVectorizer` (20,000 features, bigrams, `min_df=5`,
`features/vectorize.py`), fit on the train split only, after a stratified
train/test split.

**Why.** Fast, cheap, interpretable baseline - no GPU needed. Fitting
after splitting prevents test-set vocabulary/IDF from leaking into
training features.

## 4. Feature storage — SQLite + CSV/JSON

**Decision.** The feature store is SQLite (`feature_store/feature_store.db`).
The interim CSV stores `tokens` as a JSON-encoded string.

**Why.** SQLite needs no server and zero setup. JSON round-trips a token
list cleanly through both CSV and SQLite without adding a Parquet
dependency.

## 5. Schema contract as JSON

**Decision.** Expected columns and dtypes live in
`validation/feature_column.json`, loaded at runtime by
`validation/validate_data.py`.

**Why.** The schema is data, not code - adding or renaming a validated
column is a one-line edit.

## 6. Versioning with DVC

**Decision.** Data and model artifacts are versioned with DVC (`dvc.yaml`
DAG + `dvc.lock`); git holds only pointers/hashes.

**Why.** The dataset and artifacts total ~1.3 GB, too large for git, and a
git commit still needs to pin an exact, reproducible data version.

## 7. Four models trained, `rnn_lstm` served

**Decision.** `train_linear.py` trains `logreg` (default,
`class_weight="balanced"`) and `linear_svc` (calibrated, its own DVC
stage). `train_rnn.py` trains an RNN/LSTM from scratch on its own word
vocabulary. `train_transformer.py` fine-tunes BERT-tiny
(`google/bert_uncased_L-2_H-128_A-2`, ~4.4M params - the only checkpoint
size practical to train fully on CPU here, §10). `serving/app.py`
(FastAPI) serves `rnn_lstm`, with `model_version` and a bounded
`confidence` (`Field(ge=0.0, le=1.0)`) on every response.

| Model | macro-F1 | accuracy | ROC-AUC |
|---|---|---|---|
| **`rnn_lstm`** | **0.8918** | 0.9434 | — |
| `linear_svc` (calibrated) | 0.8652 | 0.9412 | 0.9671 |
| `bert_tiny` | 0.8519 | 0.9208 | 0.9704 |
| `logreg` | 0.8358 | 0.9073 | 0.9690 |

**Why.** `rnn_lstm` wins on macro-F1 (§8) by a clear margin, driven by
NEGATIVE-class recall (0.9324 vs. 0.70-0.91 for the others, §8) - reading
word order matters more here than `bert_tiny`'s pretrained knowledge or
`linear_svc`'s narrow accuracy edge. The root `Dockerfile` packages the API
with a scoped `serving/requirements.txt` (`torch` only, no
`transformers`/`mlflow`/`dvc`), since `rnn_lstm` needs none of them to
serve.

## 8. Macro-F1 headline metric + class weighting

**Decision.** Macro-F1, not accuracy, is the main score. All four models
weight the rare NEGATIVE class during training (`class_weight="balanced"`
for the linear models, a weighted loss for the RNN and transformer).

**Why.** The data is 86.5%/13.5% imbalanced (§2), so accuracy alone can
look good while ignoring the small class - macro-F1 can't be inflated that
way. Weighting measurably helps: NEGATIVE recall is 0.70-0.93 across the
four weighted models, versus 0.67 for `logreg` run unweighted (§10).

## 9. MLflow experiment tracking

**Decision.** Every run (`training/tracking.py`) logs its parameters,
metrics, model/confusion-matrix files, a `pip freeze` snapshot, the git
commit and dirty flag, a feature-store hash, and a hypothesis/conclusion
pair - stored locally in `mlflow.db`. `training/compare_runs.py` prints a
ranked leaderboard.

**Why.** DVC (§6) tracks which data went in; MLflow tracks which settings
and code produced a given score. Standardizing what every run logs is what
makes different models comparable at all.

## 10. Tuning alternatives considered and not adopted

**Context.** Every row is a real, measured alternative to the current
config, judged by macro-F1 (§8) - kept on record so a rejected idea isn't
retried by accident.

| Alternative | Result | Current default |
|---|---|---|
| TF-IDF: 30,000 features, trigrams | `saga` didn't converge; macro-F1 0.537 | 20,000 features, bigrams (§3) |
| TF-IDF: 24,000 features, bigrams | Still non-convergent; macro-F1 0.556 | 20,000 features, bigrams (§3) |
| RNN epochs: 3 vs. 4 vs. 6 | 3 wins every time (0.6488 vs. 0.6334 vs. 0.6438), deterministically | 3 epochs |
| Transformer input: raw `full_review` vs. `clean_review` | 0.6459 vs. 0.6461 - a wash | `clean_review` |
| `linear_svc` + `CalibratedClassifierCV` | Accuracy up, macro-F1 down (0.6225→0.6049) | Kept as a serving candidate, not selected (§7) |
| Transformer: `distilbert-base-uncased` (~66M params) | ~20+ hours/epoch extrapolated on CPU | BERT-tiny (~4.4M params, §7) |
| `logreg` without class weighting | NEGATIVE recall 0.6698, vs. 0.70-0.93 weighted (§8) | `class_weight="balanced"` |

## 11. Plain HTML UI, no framework

**Decision.** `ui/index.html` is one self-contained HTML/CSS/JS file - a
text box, an Analyze button, a result view. `serving/app.py` enables CORS
so the UI (a different local port) can call the API directly.

**Why.** One form talking to one API endpoint doesn't need a framework's
complexity - open the file and it works, nothing to install or compile.

## 12. Monitoring, drift simulation, and retraining triggers

**Decision.** Full design: [Monitoring & retraining](../monitoring.md).
Every prediction is logged to SQLite (`monitoring/prediction_log.py`); a
training-time baseline is recorded once (`monitoring/baseline.py`); a
30-review, hand-labeled modern-slang set simulates unfamiliar-vocabulary
traffic (`monitoring/simulate_drift.py`); `monitoring/monitor.py` checks
four signals against the baseline - vocabulary drift and confidence/output
drift (label-free, warn early) and macro-F1 drop (the direct signal, but
only computable with true labels) - and recommends retraining only on
measured evidence, never a fixed schedule. The monitor reports; it doesn't
retrain or redeploy automatically - `dvc repro --force train_rnn` stays a
manual step, matching this project's preference for locally-runnable tools
over heavier automation (§4, §9).

**Why.** The API returns a valid `200` and a normal-range confidence even
when predictions have drifted - nothing in the response format changes, so
catching it needs active comparison against a baseline. On the drift set,
macro-F1 falls to 0.7600 from a 0.8918 baseline, and two of the four
signals fire.

## 13. API failure handling and contract-change policy

**Decision.** A missing model file at startup fails soft: `serving/app.py`
keeps running with `_model = None`, `/health` reports `"model_not_loaded"`,
and `/predict` returns `503`, not `500`. A new contract field ships
optional first; it only becomes required in a new versioned endpoint
(e.g. `/v2/predict`).

**Why.** `503` ("temporarily unavailable") is the accurate meaning here,
versus `500`'s "something broke unexpectedly" - a caller can tell the two
apart. A required field added immediately would turn every existing
caller's next request into an unexpected `422`; optional-first is the only
change that can't break anyone already depending on the current shape.
