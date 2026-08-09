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
data/processed/train_v1.csv  +  model_store/tfidf_vectorizer_v1.pkl
        │  training/  (reserved — model training)
        ▼
      model
```

## Stages

1. **Build features** — `python -m features.build_features`
   Loads raw → drops metadata columns → combines positive+negative into
   `full_review` → de-duplicates on `full_review` + `Reviewer_Score` and drops
   rows with an empty `full_review` → cleans/tokenizes into `clean_review` +
   `tokens` → derives the `sentiment` label → writes
   `data/interim/features_clean.csv`.
   Cleaning expands contractions (`don't` / `don t` → `do not`, curly
   apostrophes normalized first) and attaches negators to the following token
   (`not good` → `not_good`, but function words such as `no one` are left
   split) — see [Decisions §10–11](design/decisions.md).
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
   `data/processed/train_v1.csv` and the fitted vectorizer
   `model_store/tfidf_vectorizer_v1.pkl`. Exposes `build_tfidf_features()` →
   `X_train_vec, X_test_vec, y_train, y_test, vectorizer`.

5. **Train** — reserved under `training/`; imports `build_tfidf_features()` from
   `features.vectorize`.

## Diagnostics (not part of the DAG)

`python -m validation.diagnose_cleaning [--limit N]` scans the interim CSV for
the known cleaning defects — negators merged with function words (`not_the`),
contractions split into `<stem> t`, stray/curly apostrophes, empty documents,
and duplicates that survived the dedupe. Each check prints `[OK]` or an
`[ISSUE]` line with counts, examples, and the fix it expects. It is a report
only: it writes nothing and always exits 0, so run it after any cleaning change.

## Schema contract

`validation/feature_column.json` is the single source of truth for the feature
columns and their dtypes. `validation/validate_data.py` loads it at runtime, so
adding or renaming a validated column is a one-line JSON edit — no code change.
See [Contributing](contributing.md) for the recipe.
