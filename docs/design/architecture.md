# Architecture guide

[← Design docs](README.md) · Related: [Decisions](decisions.md) · [Pipeline](../pipeline.md) · [Versioning](../versioning.md)

## Overview

The system is a **linear, file-based ML data pipeline**: raw reviews are cleaned
and labeled, validated against a schema contract, materialized into a feature
store, then vectorized into model-ready features. Every stage is an independent,
rebuildable module wired together by a DVC DAG.

## Components & layers

| Component        | Responsibility                                                        |
|------------------|-----------------------------------------------------------------------|
| **Data layers**  | `data/raw` (immutable source) → `data/interim` (cleaned/labeled) → `data/processed` (model-ready splits) |
| **`features/`**  | `build_features.py` (ingest → clean/tokenize → label) and `vectorize.py` (split + TF-IDF) |
| **`validation/`**| `validate_data.py` gate, driven by the `feature_column.json` schema contract |
| **`feature_store/`** | `feature_store.py` — persists features to SQLite (`feature_store.db`, table `hotel_review_features`) |
| **`model_store/`**   | Persisted model artifacts (`tfidf_vectorizer_v1.pkl`)             |
| **`training/`**  | Reserved — model trainer, consumes `build_tfidf_features()`            |
| **`serving/`, `ui/`** | Reserved — inference API and UI                                  |

## Data flow (DVC DAG)

Defined in `dvc.yaml`; run with `dvc repro`:

```
data/raw/Hotel_Reviews.csv
    │  build_features   (python -m features.build_features)
    ▼
data/interim/features_clean.csv ──► validate  (python -m validation.validate_data)   [gate, no outs]
    │  feature_store    (python -m feature_store.feature_store)
    ▼
feature_store/feature_store.db
    │  vectorize        (python -m features.vectorize)
    ▼
data/processed/train_v1.csv  +  model_store/tfidf_vectorizer_v1.pkl
```

| Stage            | Deps                                                      | Outs                                                  |
|------------------|-----------------------------------------------------------|-------------------------------------------------------|
| `build_features` | `features/build_features.py`, `data/raw/Hotel_Reviews.csv`| `data/interim/features_clean.csv`                     |
| `validate`       | interim CSV, `validate_data.py`, `feature_column.json`    | *(none — always-run gate)*                            |
| `feature_store`  | interim CSV, `feature_store.py`                           | `feature_store/feature_store.db`                      |
| `vectorize`      | `feature_store.db`, `vectorize.py`                        | `data/processed/train_v1.csv`, `tfidf_vectorizer_v1.pkl` |

## Tech stack

- **pandas** — dataframe transforms across every stage.
- **SQLite + SQLAlchemy** — the feature store; serverless and file-based.
- **scikit-learn** — `TfidfVectorizer` + `train_test_split`; `joblib` for artifact persistence.
- **DVC** — pipeline definition (`dvc.yaml`) and data/artifact versioning alongside git.

See [Decisions](decisions.md) for *why* each was chosen.

## Design principles

- **One module per stage**, runnable from the repo root as `python -m <pkg>.<module>`
  with an `if __name__ == "__main__":` entrypoint.
- **Rebuildable / idempotent stages** — outputs are a pure function of inputs, so
  `dvc repro` can rebuild any layer; `feature_store`/`vectorize` auto-regenerate
  the interim if it's missing.
- **Schema as data** — the validated column contract lives in
  `validation/feature_column.json`, not in Python.
- **Train/serve consistency** — the fitted vectorizer is versioned alongside the
  exact split it was fit on (TF-IDF is fit on the train split only).
- **Versioning via DVC** — git holds pointers/hashes; data lives in the DVC cache/remote.

## Extension points

- **Model training** → `training/`: import `build_tfidf_features()` from
  `features.vectorize`, fit a classifier, save it to `model_store/` (versioned), and
  add a `train` stage to `dvc.yaml`.
- **Serving** → `serving/`: load the vectorizer + model from `model_store/` behind an API.
- **UI** → `ui/`: front-end over the serving API.
