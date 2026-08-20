# Architecture guide

[← Design docs](README.md) · Related: [Decisions](decisions.md) · [Pipeline](../pipeline.md) · [Versioning](../versioning.md)

## Overview

There are two parts to this project.

The **training pipeline** takes raw hotel reviews and turns them into a
trained model. It runs as a series of small steps, one after another - clean
the text, check it looks right, save it somewhere, turn it into numbers, then
train a model on those numbers. Each step is its own Python file, and each
one reads what the last step produced. DVC is the tool that runs these steps
in order and only re-runs a step if something it depends on actually changed.
Every time a model gets trained, the settings and the score get logged too,
using a tool called MLflow, so it's possible to look back and see what was
tried and what happened.

The **serving part** is separate from all of that (more on why in
[Decisions §22-23](decisions.md)). It's a small API that loads one of the
four trained models - `rnn_lstm` - and lets you send it a review and get a
prediction back. There's also a simple web page on top of that API so a
person can just type something in and see the result, instead of needing to
send requests manually. The two halves connect through one shared folder,
`model_store/` - training writes files there, and serving reads one of them.

## Components & layers

| Folder / file | What it does |
|------------------|-----------------------------------------------------------------------|
| **Data folders**  | `data/raw` (the original file, never touched) → `data/interim` (cleaned up) → `data/processed` (ready to train on) |
| **`features/`**  | `build_features.py` cleans the raw data and figures out the sentiment label. `vectorize.py` splits the data and turns the text into numbers. |
| **`validation/`**| `validate_data.py` double-checks the cleaned data looks the way it's supposed to. `diagnose_cleaning.py` is a separate report you can run by hand to check for known text-cleaning problems. |
| **`feature_store/`** | Saves the cleaned data into a small local database file (SQLite - no server needed, it's just a file). |
| **`model_store/`**   | Every trained model ends up here.             |
| **`training/`**  | The four training scripts, plus `tracking.py` (logs runs to MLflow) and `compare_runs.py` (prints a leaderboard of past runs). |
| **`mlflow.db`, `mlruns/`** | Where the training logs actually live on disk. Not checked into git - it's just a record, not code. |
| **`serving/`**   | `app.py` is the API. It has a `/health` check and a `/predict` endpoint, and it currently serves `rnn_lstm`. The `Dockerfile` at the repo root packages it so it runs the same way anywhere. |
| **`ui/`**        | `index.html` - one plain web page that calls the API and shows what it says. No extra tools or setup needed to run it. |

## Data flow

The DVC pipeline (defined in `dvc.yaml`, run with `dvc repro`) stops once it
has produced trained models in `model_store/`. Serving picks up after that,
and you start it yourself - `dvc repro` doesn't do it for you:

```
data/raw/Hotel_Reviews.csv
    │  build_features   (python -m features.build_features)
    ▼
data/interim/features_clean.csv ──► validate  (python -m validation.validate_data)   [just a check, no file produced]
    │  feature_store    (python -m feature_store.feature_store)
    ▼
feature_store/feature_store.db
    │  vectorize        (python -m features.vectorize)
    ▼
data/processed/train_v1.csv + test_v1.csv  +  model_store/tfidf_vectorizer_v1.pkl
    ├─ train / train_linear_svc / train_rnn / train_transformer
    ▼
model_store/*  +  training/metrics*.json  ──▶ training/tracking.py ──▶ mlflow.db
    │
    │  (the DVC pipeline stops here - everything past this point is started by hand)
    ▼
model_store/rnn_lstm_v1.pt + rnn_lstm_v1_vocab.json
    │  serving/app.py   (uvicorn serving.app:app, or the Dockerfile)
    ▼
REST API on :8000  (/health, /predict)
    ▲
    │  request sent from the browser
    │
ui/index.html   (run separately, e.g. python -m http.server 8090)
```

| Stage            | Reads                                                      | Writes                                                  |
|------------------|-------------------------------------------------------------|-------------------------------------------------------|
| `build_features` | `features/build_features.py`, `data/raw/Hotel_Reviews.csv`| `data/interim/features_clean.csv`                     |
| `validate`       | interim CSV, `validate_data.py`, `feature_column.json`    | *(nothing - pass/fail only)*                            |
| `feature_store`  | interim CSV, `feature_store.py`                           | `feature_store/feature_store.db`                      |
| `vectorize`      | `feature_store.db`, `vectorize.py`                        | `data/processed/train_v1.csv`, `test_v1.csv`, `tfidf_vectorizer_v1.pkl` |
| `train` / `train_linear_svc` | both splits, `tfidf_vectorizer_v1.pkl`, `train_linear.py`, `tracking.py` | a trained model + its scores |
| `train_rnn`      | both splits, `train_rnn.py`, `tracking.py`                | `rnn_lstm_v1.pt` + its vocabulary file + scores               |
| `train_transformer` | both splits, `train_transformer.py`, `tracking.py`     | `bert_tiny_v1/` + scores                        |

## Tech stack

- **pandas** — loads and reshapes the data at every step.
- **SQLite + SQLAlchemy** — the feature store. Just a file, nothing to set up.
- **scikit-learn** — turns text into numbers (TF-IDF) and splits the data.
- **DVC** — runs the pipeline steps in order and keeps the data files
  versioned alongside the code.
- **PyTorch** — trains the RNN model from scratch.
- **transformers** (HuggingFace) — fine-tunes the pretrained BERT-tiny model.
- **MLflow** — logs every run's settings and scores so past results aren't lost.
- **FastAPI** — what the API is built with.

See [Decisions](decisions.md) for why each of these got picked over other options.

## Design principles

A few habits this project tries to stick to:

- **One step, one file** - each stage of the pipeline is its own script you
  can run on its own.
- **Everything is rebuildable** - if a file is missing or out of date,
  `dvc repro` can just regenerate it from what came before.
- **The schema lives in a data file, not in code** - the columns the
  pipeline expects live in `validation/feature_column.json`, so changing
  them doesn't mean touching Python.
- **Training and serving use the same setup** - the fitted TF-IDF vectorizer
  is saved together with the exact data it was fit on, so they can't quietly
  drift apart from each other.
- **DVC handles versioning** - git keeps small pointer files, the actual
  data lives in DVC's own storage.
- **Nothing important only lives in memory** - every setting and score from
  a training run gets logged, so it can be looked up later instead of relying
  on memory.

## Extension points

Where you'd go to make common changes:

- **Adding a new model** → `training/`. Load the saved splits and the fitted
  vectorizer, wrap the training call in `training.tracking.start_run(...)`
  so it gets logged, save the model to `model_store/`, and add a matching
  stage to `dvc.yaml`.
- **Changing what the API serves** → `serving/app.py`. Right now it serves
  `rnn_lstm`, and it's packaged with the `Dockerfile` at the repo root
  (§22).
- **Changing the UI** → `ui/index.html`. Plain HTML/CSS/JavaScript, nothing
  to install or build (§23).
