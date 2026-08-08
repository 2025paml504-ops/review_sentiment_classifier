# review_sentiment_classifier


A sentiment classifier for hotel reviews. A travel e-commerce platform wants to
automatically classify incoming hotel-review text by sentiment into three
classes: **NEGATIVE**, **NEUTRAL**, and **POSITIVE**.

Built on an all-open-source stack: `pandas`, `SQLAlchemy` (SQLite),
`scikit-learn`, and `DVC` for data/artifact versioning.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Get the data: `dvc pull` (if you have remote access) OR place the Kaggle CSV
# at data/raw/Hotel_Reviews.csv (see docs/dataset.md), then build everything:
.venv/bin/dvc repro
```

`dvc repro` runs the pipeline stages in dependency order and skips anything
already up-to-date.

## Repository layout

| Path             | Role                                                             |
|------------------|------------------------------------------------------------------|
| `data/raw/`      | Immutable source data (`Hotel_Reviews.csv`)                      |
| `data/interim/`  | Cleaned/labeled, rebuildable intermediate (`features_clean.csv`) |
| `data/processed/`| Model-ready splits (`train_v1.csv`)                              |
| `features/`      | Feature engineering (`build_features.py`, `vectorize.py`)        |
| `validation/`    | Data-quality checks + JSON schema contract                       |
| `feature_store/` | SQLite feature store (`feature_store.db`)                        |
| `model_store/`   | Persisted model artifacts (`tfidf_vectorizer_v1.pkl`)            |
| `training/`      | Reserved for the model trainer (consumes the TF-IDF features)    |
| `serving/`, `ui/`| Reserved for serving and UI                                      |

## Documentation

- **[Dataset](docs/dataset.md)** — source (Kaggle 515K) and the raw/interim/processed data layers.
- **[Pipeline](docs/pipeline.md)** — the DVC DAG, each stage in detail, and the schema contract.
- **[Versioning](docs/versioning.md)** — how DVC versions data/artifacts, the remote, and cutting a new version.
- **[Contributing](docs/contributing.md)** — prerequisites, code conventions, common-task recipes, and the pre-commit checklist.
- **[Design](docs/design/README.md)** — architecture guide and the decision-making guide (why cleaning, sentiment thresholds, TF-IDF, SQLite, DVC, …).

## Status / roadmap

The data → features → feature store → TF-IDF pipeline is complete. `training/`,
`serving/`, and `ui/` are reserved for the next tasks (model training, API
serving, and UI).

