# Contributing

[← Back to README](../README.md) · Related: [Pipeline](pipeline.md) · [Versioning](versioning.md)

## Prerequisites

- **Python 3.14** + `venv`
- **git**
- **DVC** (installed via `requirements.txt`)

All dependencies are open source: `pandas`, `SQLAlchemy` (SQLite), `scikit-learn`,
and `dvc`.

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
   rebuilt interim CSV — it reports the known cleaning defects (negators merged
   with function words, contractions split into `<stem> t`, stray apostrophes,
   empty documents, surviving duplicates) with counts and examples. Every check
   should print `[OK]`; a `[ISSUE]` line names the fix it expects.

## Before you commit

- [ ] `dvc repro` (or at least `python -m validation.validate_data`) passes green.
- [ ] If you touched cleaning: `python -m validation.diagnose_cleaning` reports
      `[OK]` for every check.
- [ ] Commit `dvc.lock` and any `*.dvc` pointer files **with** the code change.
- [ ] Do **not** commit files under `data/`, `feature_store/*.db`, or
      `model_store/*` — these are DVC-tracked and git-ignored.
- [ ] `dvc push` if you have access to a shared remote.
