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
already up-to-date. The pipeline now also trains four models; the last stage
(`train_transformer`) is the heaviest (~1 hour) — run `dvc repro train`
instead to stop at the fast linear baseline.

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
| `serving/`       | FastAPI REST API (`/health`, `/predict`) serving `logreg`; `Dockerfile` at repo root packages it |
| `ui/`            | Reserved — not yet built                                          |

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

The data → features → feature store → TF-IDF → training pipeline covers four
models — `logreg` (default), `linear_svc`, a recurrent net (`rnn_lstm`), and a
BERT-mini fine-tune — trained on the same splits and compared on macro-F1.
Every run is tracked in MLflow, so results are comparable and reproducible.
See [Decisions §13–20](docs/design/decisions.md) for what was built, and
[Decisions §21](docs/design/decisions.md) for the tuning experiments tried and
(where they didn't help) reverted. `serving/` now has a working REST API
([serving/README.md](serving/README.md), [Decisions §22](docs/design/decisions.md));
`ui/` remains reserved.
