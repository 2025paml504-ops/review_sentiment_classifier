# review_sentiment_classifier


A sentiment classifier for hotel reviews. A travel e-commerce platform wants to
automatically classify incoming hotel-review text by sentiment into three
classes: **NEGATIVE**, **NEUTRAL**, and **POSITIVE**.

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
already up-to-date. The pipeline trains four models; the heaviest is the
last stage, `train_transformer` (`bert_mini`, hours on CPU). Run
`dvc repro train_rnn` instead to stop after `rnn_lstm` - that's the model
`serving/` actually uses, and it's a fast stage.

Then, to run the API and the UI (needs `rnn_lstm` from the step above):

```bash
# one terminal
.venv/bin/uvicorn serving.app:app --reload --port 8000

# a second terminal
.venv/bin/python -m http.server 8090 --directory ui
```

Open `http://localhost:8090/index.html`. Details: [serving/README.md](serving/README.md).

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
(`logreg`, `linear_svc`, `rnn_lstm`, `bert_mini`) -> compared on macro-F1
-> tracked in MLflow. [Decisions §13–20](docs/design/decisions.md) (models)
-> [§21](docs/design/decisions.md) (tuning experiments, kept vs. reverted).

`serving/` and `ui/` are now both built — see [serving/README.md](serving/README.md)
and [Decisions §22–23](docs/design/decisions.md).
