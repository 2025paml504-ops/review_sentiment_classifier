# Data & artifact versioning (DVC)

[← Back to README](../README.md) · Related: [Dataset](dataset.md) · [Pipeline](pipeline.md)

Data and model artifacts are versioned with **[DVC](https://dvc.org)**, which
works alongside git: git tracks small text pointers (`dvc.yaml`, `dvc.lock`,
`*.dvc`, `.dvcignore`) while the large files (~1.3 GB) live in DVC's cache/remote
and never enter git history. The whole pipeline is defined as a DVC DAG in
`dvc.yaml` (`build_features → validate → feature_store → vectorize →
{train, train_linear_svc, train_rnn, train_transformer}`), so `dvc.lock`
records the exact content hash of every artifact — pinning a reproducible
dataset and model version to each git commit.

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

The default remote is named `localremote`. Only the **name** is git-tracked
(`.dvc/config`); the **URL is per-machine** and lives in the git-ignored
`.dvc/config.local`, set once per checkout:

```bash
dvc remote modify --local localremote url <your-path-or-url>
```

`dvc push` copies cached artifacts there; `dvc pull` restores them. Swap in a
cloud remote (S3/GCS/Azure/SSH) the same way — the rest of the workflow is
unchanged.

## Naming convention

The `_v1` suffix (`train_v1.csv`, `tfidf_vectorizer_v1.pkl`, `logreg_v1.pkl`,
`linear_svc_v1.pkl`, `rnn_lstm_v1.pt`, `bert_mini_v1/`, …) is a human-readable
label that coexists with DVC's content hashes — one artifact set per trained
model, all versioned against the same data contract. Bump to `_v2` when the
cleaning/tokenization logic, the sentiment labeling thresholds, or the
vectorizer configuration change. A fitted vectorizer must always be versioned
**alongside the exact split it was fit on**, and a trained model is only
valid with the vectorizer it was trained against.

The current **v1 contract** is: Scheme A sentiment thresholds
(`NEGATIVE < 6`, `6 ≤ NEUTRAL < 8`, `POSITIVE ≥ 8` on `Reviewer_Score`) and the
column schema in `validation/feature_column.json`.

## Version history

| Version | Change | Impact | [Decisions](design/decisions.md) sections |
|---|---|---|---|
| **v1** | Original `main`: cleaning → Scheme A thresholds → TF-IDF fit on train only. | Baseline. | §1–9 |
| **1.1** | Cleaning overhaul: contraction expansion, negation attachment (`not good` → `not_good`), earlier de-duplication, and the `diagnose_cleaning` diagnostic. | 504,731 rows. Thresholds/schema unchanged. | §10–12 |
| **1.2** | Four training stages + MLflow experiment tracking added. A negation-scope bug (6.9% of rows) and a placeholder leak into `full_review` (31% of rows) were found and fixed. Several tuning experiments were tried and measured, then rolled back when they didn't help. Also added a markdown leaderboard export (`docs/model_leaderboard.md`, one row per model's best completed run) and per-run `pip freeze` logging, closing the library-version reproducibility gap. | New model artifacts per stage; 503,446 rows after the dedup fix; data contract unchanged. | §13–20, most of §21 |
| **1.3** | Leaderboard now ranks by each model's best run, not latest; renamed to `docs/model_leaderboard.md`. Documented the artifact store location as `mlruns/`, and added a Reproducibility section. RNN epochs re-verified — 3 remains best. | Documentation and tooling only; no data-contract change. | §21 (RNN re-verification row) |

Experiment tracking (MLflow) and reproducibility are covered in
[Pipeline](pipeline.md#experiment-tracking), not here — this file is DVC's
side of versioning specifically; that one covers what happens once a stage
actually runs.
