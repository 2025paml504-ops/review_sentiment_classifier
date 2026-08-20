# review_sentiment_classifier


A sentiment classifier for hotel reviews. A travel e-commerce platform wants to
automatically classify incoming hotel-review text by sentiment into two
classes: **NEGATIVE** and **POSITIVE**.

Built on an all-open-source stack: `pandas`, `SQLAlchemy` (SQLite),
`scikit-learn`, `DVC` for data/artifact versioning, `MLflow` for experiment
tracking, and `PyTorch`/`transformers` for the recurrent and transformer models.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Get the data: `dvc pull` (if you have remote access) OR place the Kaggle CSV
# at data/raw/Hotel_Reviews.csv (see docs/dataset.md), then build everything:
.venv/bin/dvc repro
```

`dvc repro` runs the pipeline stages in dependency order and skips anything
already up-to-date. The pipeline trains four models; `train_rnn` produces
`rnn_lstm` - the model `serving/` actually uses, and a fast stage. Run
`dvc repro train_rnn` to stop there instead of running all four.

Then, to run the API and the UI (needs `rnn_lstm` from the step above):

```bash
# one terminal
.venv/bin/uvicorn serving.app:app --reload --port 8000

# a second terminal
.venv/bin/python -m http.server 8090 --directory ui
```

Open `http://localhost:8090/index.html`. Details: [serving/README.md](serving/README.md).

To see the training side - every run's parameters, metrics, and tags, not
just the served model - open MLflow's own UI:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000`. This is a separate thing from the
sentiment API above: it shows past *training* runs (`logreg`, `linear_svc`,
`rnn_lstm`, `bert_tiny`, all under the `review_sentiment` experiment), not
live predictions.

## Repository layout

| Path             | Role                                                             |
|------------------|------------------------------------------------------------------|
| `data/raw/`      | Immutable source data (`Hotel_Reviews.csv`)                      |
| `data/interim/`  | Cleaned/labeled, rebuildable intermediate (`features_clean.csv`) |
| `data/processed/`| Model-ready splits (`train_v1.csv`, `test_v1.csv`)               |
| `features/`      | Feature engineering (`build_features.py`, `vectorize.py`)        |
| `validation/`    | Data-quality checks + JSON schema contract                       |
| `feature_store/` | SQLite feature store (`feature_store.db`)                        |
| `model_store/`   | Persisted model artifacts, one set per trained model              |
| `training/`      | Four trainers, MLflow tracking (`tracking.py`), leaderboard (`compare_runs.py`) |
| `mlflow.db`, `mlruns/` | Local MLflow tracking store (git-ignored, regenerable)  |
| `serving/`       | FastAPI REST API (`/health`, `/predict`) serving `rnn_lstm`; `Dockerfile` at repo root packages it |
| `ui/`            | Static page (`index.html`) that calls the serving API and shows the result |

## Documentation

**New here? Read in this order:** this README → [Versioning](docs/versioning.md)
(what `1.3` means) → [Pipeline](docs/pipeline.md) (the stages, MLflow
tracking, and reproducibility, before running `dvc repro`) →
[Dataset](docs/dataset.md) (getting the raw CSV) →
[Model leaderboard](docs/model_leaderboard.md) (current scores) →
[Decisions](docs/design/decisions.md) (why each choice was made, read last —
this is the deep dive, not required just to run the pipeline).

- **[Dataset](docs/dataset.md)** — source (Kaggle 515K) and the raw/interim/processed data layers.
- **[Pipeline](docs/pipeline.md)** — the DVC DAG, each stage in detail, the schema contract, MLflow experiment tracking, and reproducibility (fixed seeds, dataset snapshots, logged parameters).
- **[Versioning](docs/versioning.md)** — how DVC versions data/artifacts and cutting a new version.
- **[Contributing](docs/contributing.md)** — prerequisites, code conventions, common-task recipes, and the pre-commit checklist.
- **[Design](docs/design/README.md)** — architecture guide and the decision-making guide (why cleaning, sentiment thresholds, TF-IDF, SQLite, DVC, …).

## Status / roadmap

Data -> features -> feature store -> TF-IDF -> four trained models
(`logreg`, `linear_svc`, `rnn_lstm`, `bert_tiny`) -> compared on macro-F1
-> tracked in MLflow. [Decisions §13–20](docs/design/decisions.md) (models)
-> [§21](docs/design/decisions.md) (tuning experiments, kept vs. reverted).

`serving/` and `ui/` are now both built — see [serving/README.md](serving/README.md)
and [Decisions §22–23](docs/design/decisions.md).

## Model Serving API

### Endpoint: POST /predict

**Request Schema:**

| Field | Type | Constraints |
|---|---|---|
| text | str | 1-5000 characters, must not be blank/whitespace-only |

**Response Schema:**

| Field | Type | Notes |
|---|---|---|
| sentiment | str | `NEGATIVE` or `POSITIVE` |
| confidence | float | 0.0-1.0, the winning class's probability |
| probabilities | dict[str, float] | Both classes' probabilities, always sums to ~1.0 |
| latency_ms | float | How long this request took to score, server-side |
| model_version | str | Which artifact answered (`rnn_lstm_v1`) |

**Model:** `rnn_lstm` (macro-F1 - see [Decisions §22](docs/design/decisions.md) for the full comparison against the other three trained models)
**Deployed:** local dev - not deployed to a public host

## Serving API Reflection

**1. What would happen if a new required field were added to `/predict` (e.g. a `language` field)?**
Adding it as *optional* (with a default) is a non-breaking change — existing callers keep working exactly as before, since FastAPI/Pydantic only requires fields that don't have a default. Making it *required* would be breaking: every existing caller who doesn't send it would suddenly get a `422` they weren't getting before. The safe path: add it optional first, let consumers start sending it if they want to, and only make it required in a new versioned endpoint (e.g. `/v2/predict`) if it ever truly needs to be mandatory — never change what an existing endpoint requires out from under callers already depending on it.

**2. Is returning a hard failure the right response when the model file is missing at startup?**
This project already handles this the safer way, not the naive one: `serving/app.py` catches the missing-file case at startup (`OSError`/`FileNotFoundError`), logs a warning, and keeps the app running with `_model = None` instead of crashing outright. `/health` then honestly reports `"model_not_loaded"` instead of pretending everything's fine, and `/predict` returns a `503 Service Unavailable` (not a `500`) — `503` specifically means "the service is temporarily unable to handle this, try again later," which is the accurate meaning here, versus `500`'s "something broke unexpectedly." A caller (or an uptime monitor) can tell the difference between "this API is broken" and "this API is up but not ready yet."

**3. If confidence scores looked suspiciously identical across many different inputs, what would you suspect?**
That would point at the input never actually reaching the model correctly — for example, a bug in `_encode()` silently producing the same all-padding token sequence regardless of the real text, or a stale/cached tensor being reused instead of the current request's. To check: feed two genuinely different reviews (a clearly positive one, a clearly negative one) through `/predict` and confirm the `probabilities` actually differ meaningfully; if they don't, inspect what `_encode()` actually produces for each input before it reaches the model.

**4. If a future model swap ever returned a confidence outside [0.0, 1.0], what happens end to end?**
`PredictResponse.confidence` is constrained with `Field(..., ge=0.0, le=1.0)` specifically for this. Because `/predict` is declared with `response_model=PredictResponse`, FastAPI validates the *outgoing* response against that schema, not just incoming requests. An out-of-range value would fail that validation, and the caller would see a `500` — a loud, honest failure — instead of silently receiving a nonsensical confidence number they might trust.

**5. What's still missing before this could safely serve real production traffic?**
Being honest about what's *not* built, not just what is:
- **No rate limiting or authentication.** CORS is wide open (`allow_origins=["*"]`) and there's no API key or user auth at all — appropriate for a local dev demo, not for a public endpoint.
- **No monitoring or drift detection.** Nothing logs predictions over time or tracks how confidence/accuracy might change once the API sees real, unseen traffic — this is a later phase of work (monitoring, drift, retraining triggers) that this project hasn't started yet.
- **No concurrency/load testing.** The latency numbers in `serving/README.md` are all measured sequentially, one request at a time — the API has never been tested under real concurrent request volume, and there's no request queueing or batching if it needed to handle that.
