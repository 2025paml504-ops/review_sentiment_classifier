# Contributing

[← Back to README](../README.md) · Related: [Pipeline](pipeline.md) · [Versioning](versioning.md)

## Prerequisites

- **Python 3.14** + `venv`
- **git**
- **DVC** (installed via `requirements.txt`)
- **MLflow** (installed via `requirements.txt`) — not optional, every training stage imports it.

All dependencies are open source and declared in `requirements.txt`: `pandas`,
`numpy`, `SQLAlchemy` (SQLite), `scikit-learn`, `joblib`, `dvc`, `mlflow`,
plus `kaggle` (optional download helper), and `torch`/`transformers`/
`datasets`/`accelerate`/`sentencepiece`/`tiktoken` for the transformer
fine-tune and the recurrent stage (which shares `torch`) — all imported
lazily, so the rest of the pipeline runs without them.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Conventions

Match the existing code when adding new work:

- **One module per pipeline stage**, runnable from the repo root as
  `python -m <package>.<module>` with an `if __name__ == "__main__":` entrypoint.
- **Repo-root paths** via `REPO_ROOT = Path(__file__).resolve().parents[1]` — never
  hardcode absolute paths or rely on the current working directory.
- **Every package has an `__init__.py`** (`features/`, `feature_store/`,
  `validation/`, `training/`).
- **Stages are rebuildable**: they read from a `data/` layer (or the feature
  store) and write to the next. `feature_store` and `vectorize` auto-regenerate
  the interim data if it's missing.
- **Every training run is tracked**: wrap the fit in
  `training.tracking.start_run(...)` and log parameters, metrics and artifacts
  through the yielded handle.

## Common tasks

### Add or modify a pipeline stage
1. Edit/add the module (e.g. `features/<new>.py`).
2. Add or adjust the stage in `dvc.yaml` (`cmd`, `deps`, `outs`).
3. `dvc repro` to run it and update `dvc.lock`.
4. Commit the code **and** `dvc.yaml` + `dvc.lock` together.

### Add a validated column
1. Produce the column in `features/build_features.py`.
2. Add it to `validation/feature_column.json` (a one-line edit — the schema is
   data, not code).
3. Re-run `python -m validation.validate_data` (or `dvc repro`) to confirm green.

### Change labeling / cleaning / vectorizer config
1. Make the change in the relevant module.
2. Bump the artifact version suffix (`_v1` → `_v2`) if the schema, the labeling
   scheme, or the vectorizer config changed, keeping the vectorizer paired with
   the split it was fit on. A cleaning-only change keeps the suffix.
3. Add a row to the [version history](versioning.md#version-history) table
   describing the change and its impact on downstream artifacts.
4. `dvc repro` (this re-fits the vectorizer — required whenever the cleaning
   changes the vocabulary), then commit the lock + pointer files.
   See [Versioning](versioning.md).
5. For a cleaning change, run `python -m validation.diagnose_cleaning` on the
   rebuilt interim CSV — it reports the known cleaning defects with counts and
   examples. Every check should print `[OK]`; a `[ISSUE]` line names the fix
   it expects.

### Add a model or change hyperparameters
1. Add the estimator to `MODELS` in `training/train_linear.py`, edit
   `training/train_rnn.py`, or edit `training/train_transformer.py`.
2. Log the change: hyperparameters into `params`, new scores into `metrics`,
   new files through `run.log_artifact(...)`. Anything not logged is
   invisible in the comparison.
3. Only the default model writes the DVC-tracked `training/metrics_logreg.json`; a
   comparison run writes `training/metrics_<model>.json`.
4. Run the trainer, then compare against previous runs in
   `mlflow ui --backend-store-uri sqlite:///mlflow.db` on macro-F1 before
   promoting anything to the default.
5. A new model **family** needs its own module, its own DVC stage, and its
   own `training/metrics_<family>.json` metric (`cache: false`).

## Before you commit

- [ ] `dvc repro` (or at least `python -m validation.validate_data`) passes green.
- [ ] If you touched cleaning: `python -m validation.diagnose_cleaning` reports
      `[OK]` for every check.
- [ ] If you touched training: the run shows up in `mlflow ui` with its
      parameters, metrics and artifacts.
- [ ] Commit `dvc.lock` and any `*.dvc` pointer files **with** the code change.
- [ ] Do **not** commit files under `data/`, `feature_store/*.db`,
      `model_store/*`, `mlflow.db` or `mlartifacts/` — these are DVC-tracked
      or regenerable, and git-ignored.
- [ ] `dvc push` if you have access to a shared remote.
