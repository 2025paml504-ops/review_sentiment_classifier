# Architecture guide

[← Design docs](README.md) · Related: [Decisions](decisions.md) · [Pipeline](../pipeline.md) · [Versioning](../versioning.md)

## Overview

The system is a **linear, file-based ML pipeline**: raw reviews are cleaned and
labeled, validated against a schema contract, materialized into a feature
store, vectorized into model-ready features, and used to train four models.
Every stage is an independent, rebuildable module wired together by a DVC DAG,
and every training run is recorded as an MLflow experiment run.

## Components & layers

| Component        | Responsibility                                                        |
|------------------|-----------------------------------------------------------------------|
| **Data layers**  | `data/raw` (immutable source) → `data/interim` (cleaned/labeled) → `data/processed` (model-ready splits) |
| **`features/`**  | `build_features.py` (ingest → dedupe/drop empty → clean/tokenize, incl. contraction expansion + negation attachment → label) and `vectorize.py` (split + TF-IDF) |
| **`validation/`**| `validate_data.py` gate, driven by the `feature_column.json` schema contract; `diagnose_cleaning.py` — off-DAG report on known cleaning defects |
| **`feature_store/`** | `feature_store.py` — persists features to SQLite (`feature_store.db`, table `hotel_review_features`) |
| **`model_store/`**   | Persisted model artifacts, one set per trained model             |
| **`training/`**  | Four trainers (`train_linear.py`, `train_rnn.py`, `train_transformer.py`), `tracking.py` (MLflow), `compare_runs.py` (leaderboard) |
| **`mlflow.db`, `mlruns/`** | Local MLflow tracking store — one record per run; git-ignored |
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
data/processed/train_v1.csv + test_v1.csv  +  model_store/tfidf_vectorizer_v1.pkl
    ├─ train / train_linear_svc / train_rnn / train_transformer
    ▼
model_store/*  +  training/metrics*.json  ──▶ training/tracking.py ──▶ mlflow.db
```

| Stage            | Deps                                                      | Outs                                                  |
|------------------|-------------------------------------------------------------|-------------------------------------------------------|
| `build_features` | `features/build_features.py`, `data/raw/Hotel_Reviews.csv`| `data/interim/features_clean.csv`                     |
| `validate`       | interim CSV, `validate_data.py`, `feature_column.json`    | *(none — always-run gate)*                            |
| `feature_store`  | interim CSV, `feature_store.py`                           | `feature_store/feature_store.db`                      |
| `vectorize`      | `feature_store.db`, `vectorize.py`                        | `data/processed/train_v1.csv`, `test_v1.csv`, `tfidf_vectorizer_v1.pkl` |
| `train` / `train_linear_svc` | both splits, `tfidf_vectorizer_v1.pkl`, `train_linear.py`, `tracking.py` | model + metrics JSON |
| `train_rnn`      | both splits, `train_rnn.py`, `tracking.py`                | `rnn_lstm_v1.pt` + vocab + metrics JSON               |
| `train_transformer` | both splits, `train_transformer.py`, `tracking.py`     | `bert_mini_v1/` + metrics JSON                        |

## Tech stack

- **pandas** — dataframe transforms across every stage.
- **SQLite + SQLAlchemy** — the feature store; serverless and file-based.
- **scikit-learn** — `TfidfVectorizer` + `train_test_split`; `joblib` for artifact persistence.
- **DVC** — pipeline definition (`dvc.yaml`) and data/artifact versioning alongside git.
- **PyTorch** — the recurrent model (`train_rnn`), trained from scratch.
- **transformers** — the BERT-mini fine-tune (`train_transformer`).
- **MLflow** — experiment tracking: parameters, metrics, tags and artifacts per
  training run, in the local SQLite store `mlflow.db`.

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
- **Nothing important lives only in memory** — every hyperparameter, metric and
  artifact of a training run is logged to MLflow, so a run is comparable and
  reproducible from its record alone.

## Extension points

- **A new model** → `training/`: consume the persisted splits + fitted
  vectorizer, wrap the fit in `training.tracking.start_run(...)`, save the
  artifact to `model_store/` (versioned), and add a stage to `dvc.yaml`.
- **Serving** → `serving/`: load the vectorizer + model from `model_store/` behind an API.
- **UI** → `ui/`: front-end over the serving API.
