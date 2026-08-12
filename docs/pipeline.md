# Pipeline

[← Back to README](../README.md) · Related: [Dataset](dataset.md) · [Versioning](versioning.md) · [Contributing](contributing.md)

The pipeline is defined as a DVC DAG in `dvc.yaml` and runs end-to-end with
`dvc repro`. Each stage is also a standalone module runnable from the repo root
as `python -m <package>.<module>`.

## Flow

```
data/raw/Hotel_Reviews.csv
        │  features.build_features   (drop cols → full_review → dedupe/drop empty → clean/tokenize → label sentiment)
        ▼
data/interim/features_clean.csv
        │  validation.validate_data  (schema / placeholders / score floor / sentiment)
        │  feature_store
        ▼
feature_store/feature_store.db  (table: hotel_review_features)
        │  features.vectorize       (stratified split, TF-IDF fit on train only)
        ▼
data/processed/train_v1.csv + test_v1.csv  +  model_store/tfidf_vectorizer_v1.pkl
        ├─ train              → model_store/logreg_v1.pkl          + training/metrics_logreg.json
        ├─ train_linear_svc   → model_store/linear_svc_v1.pkl      + training/metrics_linear_svc.json
        ├─ train_rnn          → model_store/rnn_lstm_v1.pt         + training/metrics_rnn_lstm.json
        └─ train_transformer  → model_store/bert_mini_v1/          + training/metrics_transformer.json

all four training stages ──▶ training/tracking.py ──▶ mlflow.db + mlruns/
```

## Stages

1. **Build features** — `python -m features.build_features`
   Loads raw → drops metadata columns → combines positive+negative into
   `full_review` → de-duplicates on `full_review` + `Reviewer_Score` and drops
   rows with an empty `full_review` → cleans/tokenizes into `clean_review` +
   `tokens` → derives the `sentiment` label → writes
   `data/interim/features_clean.csv`.
   Cleaning expands contractions (`don't` / `don t` → `do not`, curly
   apostrophes normalized first) and attaches negators to the following
   content word (`not good` → `not_good`, `not the best` → `not_best`), while
   fixed phrases like `no one` are left split — see
   [Decisions §10, §21](design/decisions.md). The `No Positive`/`No Negative`
   placeholder is stripped at the source, before it can leak into any text
   column — see [Decisions §1](design/decisions.md).
   Rows whose `clean_review` comes out empty (text was only placeholders or
   non-letters) are dropped after cleaning.

2. **Validate** — `python -m validation.validate_data`
   Schema check (driven by `validation/feature_column.json`), placeholder
   detection (`No Positive` / `No Negative`), score-floor sanity check, empty
   `full_review` warning, and sentiment-label validity + class distribution.
   Exits non-zero on hard errors.

3. **Load feature store** — `python -m feature_store.feature_store`
   Reads the interim CSV (regenerating it if missing) and writes it to the
   SQLite table `hotel_review_features` in `feature_store/feature_store.db`.

4. **Vectorize** — `python -m features.vectorize`
   Stratified train/test split (`clean_review` = X, `sentiment` = y), TF-IDF
   **fit on the train split only** (no leakage), then writes
   `data/processed/train_v1.csv`, `test_v1.csv`, and the fitted vectorizer
   `model_store/tfidf_vectorizer_v1.pkl`. Exposes `build_tfidf_features()` →
   `X_train_vec, X_test_vec, y_train, y_test, vectorizer`.

5. **Train** — `python -m training.train_linear [--model logreg|linear_svc]`
   Loads the persisted splits and the already-fitted vectorizer (`transform`
   only, never re-fit), trains the classifier, evaluates on the test split,
   and writes the model plus a metrics JSON. `logreg` (default) and
   `linear_svc` are a like-for-like comparison pair.

6. **Train RNN** — `python -m training.train_rnn [--arch rnn_lstm|rnn_bilstm|rnn_simple]`
   Reads the same splits but **not** the TF-IDF vectorizer — a recurrent net
   needs an ordered sequence of token ids, so it builds its own word index
   from the train split and trains an Embedding + LSTM stack in PyTorch.

7. **Train transformer** — `python -m training.train_transformer`
   Fine-tunes a pretrained BERT-mini encoder on the same splits, so its
   macro-F1 is directly comparable to the other three.




## Schema contract

`validation/feature_column.json` is the single source of truth for the feature
columns and their dtypes. `validation/validate_data.py` loads it at runtime, so
adding or renaming a validated column is a one-line JSON edit — no code change.
See [Contributing](contributing.md) for the recipe.

## Diagnostics (not part of the DAG)

`python -m validation.diagnose_cleaning [--limit N]` scans the interim CSV for
the known cleaning defects — negators merged with function words (`not_the`),
contractions split into `<stem> t`, stray/curly apostrophes, empty documents,
and duplicates that survived the dedupe. Each check prints `[OK]` or an
`[ISSUE]` line with counts, examples, and the fix it expects. It is a report
only: it writes nothing and always exits 0, so run it after any cleaning change.

## Experiment tracking

Every training run also logs to **MLflow** (`training/tracking.py`):
`start_run(...)` auto-attaches the git commit/dirty state, a feature-store
snapshot, `dvc.lock`, and a `pip freeze` snapshot before the trainer logs its
own params/metrics — stored in `mlflow.db` / `mlruns/` (git-ignored,
regenerable). Model versioning is three layers: `dvc.lock` pins each model's
exact hash per commit, the `_v1` filename gets overwritten every retrain, and
MLflow keeps an immutable per-run copy for `logreg`/`linear_svc` only — the
RNN and transformer have no such copy, so `dvc.lock` is their only way back to
an old version. Compare runs with `mlflow ui --backend-store-uri
sqlite:///mlflow.db` or `python -m training.compare_runs` (`--md` writes
[`docs/model_leaderboard.md`](model_leaderboard.md), ranked by each model's
best run, not latest). See [Decisions §16](design/decisions.md).

## Reproducibility

Same inputs and same code give the same outputs here: `RANDOM_STATE = 42`
(`features/vectorize.py`) is threaded through the split, every sklearn
estimator, `torch.manual_seed` plus the RNN's `DataLoader` generator, and
HuggingFace's `set_seed()` / `Trainer(seed=...)` — verified directly, since
re-running the RNN reproduces identical loss values at a given epoch.
`feature_store.snapshot()` fingerprints the data (sha256, row count, label
distribution) so "which data" is a hash, not a mutable path, and every
hyperparameter is logged per run, not just the saved model. The one real gap:
`requirements.txt` pins no versions, so a fresh install can silently diverge —
the `pip freeze` snapshot above reveals that drift after the fact, it does not
prevent it. See [Decisions §21](design/decisions.md).
