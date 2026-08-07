# Data & artifact versioning (DVC)

[← Back to README](../README.md) · Related: [Dataset](dataset.md) · [Pipeline](pipeline.md)

Data and model artifacts are versioned with **[DVC](https://dvc.org)**, which
works alongside git: git tracks small text pointers (`dvc.yaml`, `dvc.lock`,
`*.dvc`, `.dvcignore`) while the large files (~1.3 GB) live in DVC's cache/remote
and never enter git history. The whole pipeline is defined as a DVC DAG in
`dvc.yaml` (`build_features → validate → feature_store → vectorize`), so
`dvc.lock` records the exact content hash of every artifact — pinning a
reproducible dataset version to each git commit.

## Everyday workflow

```bash
dvc repro          # rebuild any stage whose deps changed (no-op if up to date)
dvc dag            # show the pipeline graph
dvc checkout       # restore tracked data/artifacts from the cache
dvc add data/raw/Hotel_Reviews.csv   # re-track after a new raw drop
dvc push / dvc pull                  # sync to the remote
```

## Cutting a new version

Change code/config, run `dvc repro`, then
`git add dvc.lock **/*.dvc && git commit` — that commit *is* the version, and
`git checkout <rev> && dvc checkout` restores the exact data behind it.

## Remote

A default **local** remote named `localremote` is configured in `.dvc/config`:

```
/Users/sonalgupta/dvc-remotes/review_sentiment_classifier
```

`dvc push` copies cached artifacts there; `dvc pull` restores them. This path is
**machine-local** — on another machine either re-point it:

```bash
dvc remote modify localremote url <your-path-or-url>
```

or swap in a cloud remote (S3/GCS/Azure/SSH) with the same command — the rest of
the workflow is unchanged.

## Naming convention

The `_v1` suffix (`train_v1.csv`, `tfidf_vectorizer_v1.pkl`) is a human-readable
label that coexists with DVC's content hashes. Bump to `_v2` when the
cleaning/tokenization logic, the sentiment labeling thresholds, or the vectorizer
configuration change. A fitted vectorizer must always be versioned **alongside
the exact split it was fit on** so training and serving use the same
vocabulary/IDF.

The current **v1 contract** is: Scheme A sentiment thresholds
(`NEGATIVE < 6`, `6 ≤ NEUTRAL < 8`, `POSITIVE ≥ 8` on `Reviewer_Score`) and the
column schema in `validation/feature_column.json`.
