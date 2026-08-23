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
.venv/bin/python3 -m http.server 8090 --directory ui
```

Open `http://localhost:8090/index.html`. Details: [serving/README.md](serving/README.md).

Alongside that, `http://127.0.0.1:8000/docs` (Swagger UI) gives an
interactive, always-accurate view of the API itself - generated directly
from `serving/app.py`, useful for testing requests/responses without
needing `curl` or the UI page.

To see the training side - every run's parameters, metrics, and tags, not
just the served model - open MLflow's own UI:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000`. This is a separate thing from the
sentiment API above: it shows past *training* runs (`logreg`, `linear_svc`,
`rnn_lstm`, `bert_tiny`, all under the `review_sentiment` experiment), not
live predictions.

Every prediction the API serves is logged automatically. To check whether
that traffic still looks like the training data, or a hand-labeled
modern-slang set built to simulate drift:

```bash
.venv/bin/python3 -m monitoring.baseline          # record what "normal" looks like (run once, or after any retrain)
.venv/bin/python3 -m monitoring.simulate_drift    # score a hand-labeled, modern-slang review set
.venv/bin/python3 -m monitoring.monitor --source drift_simulation   # check for drift + a retraining recommendation
.venv/bin/python3 -m monitoring.monitor --source api                 # same check, against real logged traffic
.venv/bin/python3 -c "from monitoring import prediction_log; print(prediction_log.read_predictions().to_string())"  # inspect the raw log
```

Details, the four retraining-trigger signals, and the measured results:
[Monitoring & retraining](docs/monitoring.md) and
[Decisions §12](docs/design/decisions.md).

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
| `monitoring/`    | Prediction logging, drift simulation, and retraining-trigger checks (`prediction_log.py`, `baseline.py`, `simulate_drift.py`, `monitor.py`) |

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
- **[Monitoring & retraining](docs/monitoring.md)** — prediction logging, the drift simulation and its measured results, and the four retraining-trigger signals.
- **[Contributing](docs/contributing.md)** — prerequisites, code conventions, common-task recipes, and the pre-commit checklist.
- **[Design](docs/design/README.md)** — architecture guide and the decision-making guide (why cleaning, sentiment thresholds, TF-IDF, SQLite, DVC, …).

## Status / roadmap

Data -> features -> feature store -> TF-IDF -> four trained models
(`logreg`, `linear_svc`, `rnn_lstm`, `bert_tiny`) -> compared on macro-F1
-> tracked in MLflow. [Decisions §7](docs/design/decisions.md) (models)
-> [§10](docs/design/decisions.md) (tuning alternatives considered and not adopted).

`serving/` and `ui/` are now both built — see [serving/README.md](serving/README.md)
and [Decisions §7, §11](docs/design/decisions.md).

Prediction logging, a drift simulation, and retraining-trigger checks are
also built — see [Monitoring & retraining](docs/monitoring.md) and
[Decisions §12](docs/design/decisions.md).

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

**Model:** `rnn_lstm` (macro-F1 - see [Decisions §7](docs/design/decisions.md) for the full comparison against the other three trained models)
**Deployed:** local dev - not deployed to a public host
**Interactive docs:** `http://127.0.0.1:8000/docs` (Swagger UI) or `/redoc` - generated directly from `serving/app.py`'s Pydantic models, so it's always accurate to the actual code, not hand-written documentation that can drift out of sync

## Serving API Reflection

**1. What would happen if a new required field were added to `/predict` (e.g. a `language` field)?**
Add it as *optional*, with a default. That's non-breaking — FastAPI/Pydantic only enforces fields with no default, so every existing caller keeps working. Making it required immediately would break them: anyone still sending the old request shape would start getting `422`s they never got before. Optional first, mandatory later (in a new versioned endpoint like `/v2/predict`) if it truly needs to be required — never change what an existing endpoint expects out from under callers already depending on it. Recorded as policy in [Decisions §13](docs/design/decisions.md).

**2. Is returning a hard failure the right response when the model file is missing at startup?**
A hard crash was considered and rejected. `serving/app.py` catches the missing-file case at startup (`OSError`/`FileNotFoundError`), logs a warning, and keeps running with `_model = None`. `/health` reports `"model_not_loaded"` instead of a false "all good." `/predict` returns `503`, not `500` — `503` means "temporarily can't handle this, try again," which is accurate here; `500` means "something broke unexpectedly," which isn't. That lets a caller, or an uptime monitor, tell "the API is broken" apart from "the API is up but not ready yet." Full rationale: [Decisions §13](docs/design/decisions.md).

**3. If confidence scores looked suspiciously identical across many different inputs, what would you suspect?**
The input isn't reaching the model correctly. Possible causes: a bug in `_encode()` producing the same all-padding sequence regardless of the real text, or a stale tensor being reused across requests. To check: feed two clearly different reviews (one positive, one negative) through `/predict` and confirm `probabilities` actually moves. If it doesn't, inspect what `_encode()` outputs for each input before it reaches the model.

**4. If a future model swap ever returned a confidence outside [0.0, 1.0], what happens end to end?**
`PredictResponse.confidence` is constrained with `Field(..., ge=0.0, le=1.0)` for exactly this case. `/predict` uses `response_model=PredictResponse`, so FastAPI validates the *outgoing* response too, not just incoming requests. An out-of-range value fails that validation, and the caller sees a `500` — a loud failure, not a confidence number that looks valid but isn't.

**5. What's still missing before this could safely serve real production traffic?**
What's not built, not just what is:
- **No rate limiting or authentication.** CORS is wide open (`allow_origins=["*"]`), no API key or user auth. Fine for local dev, not for a public endpoint.
- **No automated retraining pipeline.** [Monitoring](docs/monitoring.md) logs every prediction and can flag when retraining looks justified, but nothing acts on that automatically. Retraining is still a manual `dvc repro --force train_rnn` (see [Decisions §12](docs/design/decisions.md) for why).
- **No concurrency/load testing.** Every latency number in `serving/README.md` came from sequential, one-at-a-time requests. Untested under real concurrent load, with no request queueing or batching.
