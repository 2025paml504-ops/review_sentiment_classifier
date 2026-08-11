# Decision-making guide

[← Design docs](README.md) · Related: [Architecture](architecture.md) · [Pipeline](../pipeline.md) · [Versioning](../versioning.md)

Lightweight ADR-style records of the key choices behind the pipeline. Each entry
gives the **context**, the **decision**, the **rationale**, and the
**alternatives** considered, so future changes can be made deliberately.

---

## 1. Text cleaning & tokenization

**Context.** Reviews are free text with a positive and a negative half; the
Booking dataset inserts placeholder strings (`No Positive` / `No Negative`) when
a half is blank.

**Decision.** In `features/build_features.py`: lowercase → normalize curly
apostrophes (`U+2019`) to ASCII → expand contractions → strip the
`no positive` / `no negative` placeholder phrases → replace every non-letter
(digits, punctuation, specials) with a space → collapse whitespace → tokenize by
whitespace split → attach negators (see §10).

**Rationale.** Dependency-light (stdlib `re` only), keeps clean alphabetic tokens
that suit a bag-of-words / TF-IDF model, and removes dataset-artifact noise that
would otherwise become features. Contractions are expanded **before** the
non-letter strip, otherwise `don't` would shatter into `don` + `t` and the
negation would be lost.

Rows are de-duplicated before cleaning (§11), and rows whose `full_review` or
`clean_review` comes out empty are dropped.

**Alternatives.** Stopword-list removal — skipped as a separate step (adds an
NLTK dependency/download; TF-IDF's `idf` already down-weights common words).
`NEGATION_SKIP_WORDS` is not a stopword list: those words stay as tokens, they
are only excluded from negator merging (§10). Lemmatization/stemming — deferred
as unnecessary for a first baseline.

## 2. Combine positive + negative into `full_review`

**Context.** Each row has two text fields but we want a single sentiment target.

**Decision.** Concatenate `Positive_Review` + `Negative_Review` into one
`full_review` field (then clean into `clean_review`).

**Rationale.** One document per review is the natural unit for a single
per-review sentiment label and a single TF-IDF vector.

## 3. Sentiment labeling — Scheme A

**Context.** The dataset has a numeric `Reviewer_Score` (2.5–10.0) but no
explicit sentiment label.

**Decision.** Derive a 3-class label by thresholding the score (**Scheme A**):
`NEGATIVE < 6`, `6 ≤ NEUTRAL < 8`, `POSITIVE ≥ 8` (`features/build_features.py`).

**Rationale.** Produces the least-imbalanced split observed (~10% / 25% / 65%),
leaving a large-enough NEGATIVE class to learn.

**Alternatives.** Schemes B/C (higher cutoffs) left <5% negatives — too few to
train a usable NEGATIVE class. Thresholds are single-source constants, so a
future scheme is a one-line change (bump the artifact version — see
[Versioning](../versioning.md)).

## 4. TF-IDF over embeddings

**Context.** Need numeric features from `clean_review` for a classifier.

**Decision.** `TfidfVectorizer` with `max_features=20000`, `ngram_range=(1,2)`,
`min_df=5`, `sublinear_tf=True` (`features/vectorize.py`).

**Rationale.** A strong, cheap, interpretable classical baseline — no GPU, fast to
fit on ~515k rows, and easy to reason about. Bigrams + `min_df` capture short
phrases while pruning rare noise.

**Alternatives.** Transformer embeddings (e.g. sentence-transformers) — heavier
(model download, compute) and deferred until the baseline is established.

## 5. Fit on train only + stratified split

**Context.** Vectorizing before splitting would leak test-set vocabulary/IDF into
training features.

**Decision.** `train_test_split(..., stratify=y)` first, then `fit_transform` the
vectorizer on **train only** and `transform` the test split.

**Rationale.** Prevents leakage (honest evaluation) and stratification preserves
the imbalanced class ratios in both splits.

## 6. Feature store = SQLite

**Context.** Need somewhere to materialize features; MySQL was requested but no
server was available locally.

**Decision.** SQLite via SQLAlchemy (`feature_store/feature_store.db`, table
`hotel_review_features`).

**Rationale.** Serverless and file-based (zero setup), browsable in PyCharm's
Database tool, and one connection-string away from MySQL if we migrate later.

**Alternatives.** MySQL — needs a running server (none available); Docker/Homebrew
setup was heavier than warranted for this stage.

## 7. Interim as CSV, tokens as JSON string

**Context.** Need a rebuildable intermediate artifact carrying a token **list** per row.

**Decision.** Write `data/interim/features_clean.csv` (CSV) and store `tokens` as a
JSON-encoded string.

**Rationale.** No new dependency (`pyarrow` is not installed), and a JSON string
round-trips cleanly through both CSV and SQLite.

**Alternatives.** Parquet — preserves list columns natively but adds a `pyarrow`
dependency; not worth it here.

## 8. Schema contract as JSON

**Context.** Validation needs to know the expected columns and dtypes.

**Decision.** Externalize the contract to `validation/feature_column.json`, loaded
at runtime by `validation/validate_data.py`.

**Rationale.** The schema is *data, not code* — adding or renaming a validated
column is a one-line JSON edit with no code change.

## 9. Versioning with DVC

**Context.** Datasets/artifacts total ~1.3 GB — too large for git — and versions
must be reproducible.

**Decision.** Track data/artifacts with DVC (`dvc.yaml` DAG + `dvc.lock`), backed
by a local remote; git holds only pointers/hashes.

**Rationale.** Reproducible (`dvc repro`), git-friendly, and fully open source; a
git commit pins an exact dataset version.

**Alternatives.** Filename-only `_vN` convention — human-readable but not
reproducible or content-addressed; kept as a *label* layered on top of DVC.

## 10. Contraction expansion & negation attachment

**Context.**  Contractions broke into junk tokens and negated praise was lost.

**Decision.** In `features/build_features.py` (added 2026-08-09): Normalize apostrophes, expand contractions, and attach negators to following sentiment words.

**Rationale.** Stdlib‑only, preserves negated sentiment as explicit features.

**Alternatives.** Scope tagging or bigram reliance — heavier and less reliable.

## 11. De-duplicate before cleaning

**Context.** Identical reviews bias classes and waste cleaning work.

**Decision.** Drop duplicates on `[full_review, Reviewer_Score]` immediately
after `combine_reviews()` — i.e. **before** cleaning/tokenization — and
`reset_index(drop=True)`.

**Rationale.** Saves regex passes, prevents leakage, keeps distinct scores.

**Alternatives.** De-duplicating on `clean_review` after cleaning — collapses
slightly more rows (two raw variants can clean to the same string) but costs a
full cleaning pass over the duplicates; not worth it.

## 12. Cleaning defects detected in code

**Context.** Manual samples spotted cleaning issues but missed how often they occurred.

**Decision.** `validation/diagnose_cleaning.py` (added 2026-08-09) scans the
interim CSV for known defects and reports counts/examples. Kept off the DAG,
since a tunable text heuristic shouldn't be able to block a build.

**Rationale.** Turns "does this fix still work" into a count instead of a
guess. Found a real defect the samples missed: 48 placeholder-only reviews
cleaning down to nothing, now dropped in `build_features()`.

## 13. Baseline model — linear classifier on TF-IDF

**Context.** ~504k documents × 20,000 sparse TF-IDF features, three classes at
roughly 10 / 25 / 65 (§3). Need a first model that is cheap to fit, honest to
evaluate, and interpretable.

**Decision.** `training/train_linear.py` (added 2026-08-10) trains
`LogisticRegression(solver="saga", class_weight="balanced", max_iter=1000,
tol=1e-3)` as the default and persists `model_store/logreg_v1.pkl`.
`LinearSVC(class_weight="balanced")` is available via `--model linear_svc` as a
comparison run.

**Rationale.** Linear models are the right fit for high-dimensional sparse text:
they train in minutes on CPU, and the per-class coefficients are directly
readable (you can inspect which `not_*` tokens drive NEGATIVE).
`class_weight="balanced"` is what stops the model collapsing onto POSITIVE.
`tol=1e-3` because `saga` does not reach the default tolerance on this matrix
within a sane iteration budget, and the extra precision does not change the
ranking. Measured on the full test split (100,947 rows):

| Model | macro-F1 | accuracy | NEGATIVE recall |
|---|---|---|---|
| LogisticRegression | 0.618 | 0.693 | 0.732 |
| LinearSVC | 0.620 | 0.718 | 0.635 |

Essentially tied on macro-F1; LogisticRegression is kept as the default because
it recovers noticeably more of the NEGATIVE class — the class the product cares
about — and it exposes `predict_proba`, which serving will want for a confidence
score.

**Alternatives.** `MultinomialNB` — useful as a seconds-long sanity floor, too
weak to ship. Tree ensembles / gradient boosting — a poor match for 20k sparse
features and far slower. Transformer fine-tuning — deferred until the classical
baseline is established (§4).

**Leakage.** The trainer loads the vectorizer fitted in the `vectorize` stage and
only calls `transform`; it never re-fits. That is also why `vectorize` now
persists `test_v1.csv` — so training depends on artifacts rather than on
re-running the split.

## 14. Headline metric — macro-F1, not accuracy

**Context.** The classes are imbalanced ~10 / 25 / 65, so a model that always
predicts POSITIVE already scores ~65% accuracy.

**Decision.** Report **macro-F1** as the headline in `training/metrics_logreg.json`,
alongside per-class precision/recall/F1 and the full confusion matrix; accuracy
and weighted-F1 are recorded but are not the target. The file is registered as a
DVC `metrics:` output with `cache: false`, so it is git-tracked and
`dvc metrics diff` shows the change between commits.

**Rationale.** Macro-F1 weights all three classes equally, so improvements on the
small NEGATIVE class actually move the number. The confusion matrix is kept
because most of the remaining error is NEUTRAL↔POSITIVE bleed — expected, given
the 8.0 cutoff in §3 is an arbitrary line through a continuous score.

**Alternatives.** Accuracy — rejected as misleading under this imbalance.
Weighted-F1 — recorded, but it inherits the same majority-class bias. ROC-AUC —
not the headline, but **recorded as a secondary metric since 2026-08-10**
(`roc_auc_macro` / `roc_auc_weighted`, one-vs-rest); see §18 for why, and how
`LinearSVC`'s missing `predict_proba` is handled.

## 15. Transformer fine-tune as a pipeline stage

**Context.** TF-IDF cannot see word order or context: `not_good` only works
because §10 hand-builds it, and the model has no notion that "the room was
anything but clean" is negative. A pretrained encoder does. §4 deferred this
until the classical baseline existed — it now does (§13), so the comparison is
meaningful.

**Decision.** `training/train_transformer.py` (added 2026-08-10) fine-tunes a
pretrained encoder (overridable with `--base-model`) via the HuggingFace
`Trainer`, on the **same** `train_v1.csv` / `test_v1.csv` splits, writing
`model_store/bert_mini_v1/` and `training/metrics_transformer.json`. It is wired
into `dvc.yaml` as the `train_transformer` stage, defined exactly like `train`:
it depends on the two split CSVs plus its own script, declares the model
directory as an `out` and the metrics JSON as a `cache: false` metric. Its
dependencies (`torch`, `transformers`, `datasets`, `accelerate`) are declared in
`requirements.txt` and still imported lazily.

**Rationale.**

- *Same splits, same metric shape.* Reusing the persisted splits means the
  transformer's macro-F1 is comparable to the baseline's 0.618 with no caveats
  about a different sample; the metrics JSON has an identical structure, so the
  two files diff directly.
- *In the DAG, but skippable.* Making it a stage means the fine-tune is
  reproducible and its metrics are diffable with `dvc metrics diff`, exactly
  like the baseline. The cost is that a plain `dvc repro` now includes a job
  that is hours long and needs ~2 GB of `torch`; anyone without a GPU or the
  extras should run `dvc repro train` to stop at the baseline, or smoke-test the
  stage standalone with `--limit`.
- *Lazy imports.* `torch` / `transformers` / `datasets` are imported inside the
  functions that need them, so the module can be imported (and linted), and the
  other four stages run, in an environment where they are absent.
- *Checkpoints outside the output.* The HuggingFace `Trainer` writes to
  `model_store/bert_mini_v1_checkpoints/`, not inside `bert_mini_v1/`, so the
  DVC out stays a clean, hash-stable model directory.
- *A directory, not a `.pkl`.* Unlike every other artifact in `model_store/`,
  the fine-tuned encoder is saved with `save_pretrained()` as a **directory**
  (`config.json`, `model.safetensors`, tokenizer files), so the naming
  convention deliberately breaks pattern: `bert_mini_v1/` has no extension.
  The weights, tokenizer and config must travel together, `safetensors` loads
  faster and executes no arbitrary code on load (unlike pickle), and
  `from_pretrained()` stays the portable load path instead of a pickle tied to
  the exact `torch`/`transformers` version. DVC handles directory outs natively
  (a `.dir` entry in `dvc.lock`), so `push`/`pull`/`checkout` behave exactly as
  they do for the `.pkl` artifacts.
- *A small encoder over BERT-base.* The default is
  `google/bert_uncased_L-4_H-256_A-4` (BERT-mini): a fraction of the size and
  runtime at a modest quality cost — the right first pass at this corpus size.
  DistilBERT was the original default, which is where the old `distilbert_v1`
  artifact name came from; the artifact is now named after what is actually
  trained (see §19).
- *`max_length=256`, 2 epochs, lr 2e-5.* Standard fine-tuning defaults; reviews
  are short, so 256 word-pieces truncates only a small tail.
- *Macro-F1 selects the best checkpoint* (`metric_for_best_model`), for the same
  reason it is the headline metric in §14.
- *Subsampled runs skip the metrics write*, so a smoke test can never be
  mistaken for a scored result — the same guard as §13's trainer. The DVC stage
  therefore always invokes the module without `--limit`/`--subsample`.

**Known caveat.** The persisted splits carry `clean_review`, which is lowercased,
stripped of punctuation and has negators glued together (`not_good`). That is
the *wrong* input distribution for an encoder pretrained on natural text, and it
will understate the transformer's ceiling. Feeding the raw `full_review` for the
same rows is the obvious follow-up; it needs the splits to carry the raw text
(or a join back to the interim CSV) and is deliberately left out of this change.

**Alternatives.** Frozen sentence-transformer embeddings + a linear head —
cheaper, but usually well below a full fine-tune. A hosted LLM API — rejected:
not open source, per-call cost, and no reproducible artifact to version.

## 16. MLflow for experiment tracking

**Context.** Up to §15 a training run left behind exactly two things: a pickle
(or a model directory) and a metrics JSON. Both are *overwritten* on the next
run. Nothing recorded which hyperparameters produced a score, which commit the
code was at, or which data version was consumed — so runs could not be compared
and a past result could not be reproduced. Renaming files (`logreg_v2_final.pkl`)
is the usual workaround and is not a versioning system.

**Decision.** `training/tracking.py` (added 2026-08-10) wraps both trainers in an
MLflow run. Every run logs its **parameters** (all estimator hyperparameters or
fine-tune settings, `random_state`, `limit`, the split file names),
**metrics** (macro-F1, weighted-F1, accuracy and the flattened per-class
scores), **tags** (`model_type`, `stage`, `framework`, `git_commit`,
`smoke_test`) and **artifacts** (the model, the vectorizer, the feature schema,
the confusion matrix, the metrics JSON, and `dvc.lock`). Runs go to the local
SQLite store `mlflow.db` (artifacts under `mlartifacts/`) in the experiment
`review_sentiment`. `mlflow` is a
required dependency in `requirements.txt` and `training/tracking.py` is a
declared dep of both DVC training stages, so tracking is part of the pipeline
rather than something a run can skip.

**Rationale.**

- *Tracking complements DVC, it does not replace it.* DVC answers "which bytes
  went in and came out of this stage"; MLflow answers "which knobs were turned
  and what did they score". The bridge between them is `dvc.lock`, logged as the
  run's dataset reference — so a run points at the exact content hashes of the
  splits it consumed, not merely at a file name.
- *Local SQLite store, no server.* `mlflow.db` needs no infrastructure, works
  offline, and `mlflow ui --backend-store-uri sqlite:///mlflow.db` reads it in
  place. The older `./mlruns` *filesystem* store was the obvious choice, but
  mlflow 3.15 put it in maintenance mode and raises `MlflowException` unless
  `MLFLOW_ALLOW_FILE_STORE=true`; opting out of a deprecated backend is worse
  than moving to the supported one, and SQLite is a single file, so it stays as
  easy to hand over as a directory. `MLFLOW_TRACKING_URI` moves it to a
  server later without touching the trainers, and `MLFLOW_EXPERIMENT_NAME`
  renames the experiment.
- *The store is git-ignored.* It is a regenerable record store containing model
  binaries; the same rule as `model_store/` — git holds pointers and text only.
  The scored metrics remain git-tracked via the DVC `metrics:` JSONs, so
  `dvc metrics diff` still works on a fresh clone.
- *A first-class dependency, not an add-on.* `mlflow` is imported at module
  level like `pandas`, and the tracking module is a declared stage dep, so
  changing it invalidates the training stages. An untracked run is a run nobody
  can reproduce, so the pipeline fails loudly rather than quietly producing
  one — the opposite of an optional, silently-skipped integration.
- *One experiment for all models.* Linear baseline, LinearSVC and DistilBERT
  runs land in the same experiment, so the UI ranks them against each other on
  macro-F1; the `model_type` and `framework` tags keep them distinguishable.
- *Smoke tests are tagged, not hidden.* `--limit`/`--subsample` runs already skip
  the metrics write (§13, §15); in MLflow they are kept but tagged
  `smoke_test=true`, so they are filterable rather than silently missing.
- *Nested metrics are flattened.* MLflow metrics must be scalars, so
  `per_class.NEGATIVE.recall` is logged as a flat key and the confusion matrix
  travels as a JSON artifact instead.

**Reproducing a run.** Read the run's `git_commit` tag → check that commit out →
`dvc checkout` (the logged `dvc.lock` pins the split hashes) → re-run the stage
with the logged parameters. Fixed seeds (`random_state=42` throughout) make the
result identical.

**Alternatives.** Weights & Biases / Neptune — hosted, account-gated, and the
project is deliberately all-open-source and offline-capable. DVC experiments
(`dvc exp`) — good for parameter sweeps tied to the DAG, but no run-comparison
UI and no artifact-per-run store; it would still leave hyperparameters
undocumented. A hand-rolled CSV of runs — what MLflow already is, minus the UI,
the artifact store and the schema.

## 17. Class imbalance — cost-sensitive loss, not resampling

**Context.** The label distribution is ~10% NEGATIVE / 25% NEUTRAL / 65%
POSITIVE (§3). A model that always predicts POSITIVE scores ~65% accuracy while
being useless, and NEGATIVE — the business-relevant class — is the one most
likely to be ignored by an unweighted objective. The split is stratified (§5),
which *preserves* the imbalance for honest evaluation but does nothing to
correct training.

**Decision.** Both model families correct the imbalance in the **loss**, never in
the data:

- `training/train_linear.py` builds `LogisticRegression` and `LinearSVC` with
  `class_weight="balanced"` (§13).
- `training/train_transformer.py` **does not yet** do so: it runs the stock
  unweighted `cross_entropy`. The weighted `Trainer` subclass described here was
  planned but never landed in the code, so the run now logs
  `model.class_weight="none"` (§18) to make the gap explicit instead of implied.
  Closing it is the open follow-up on this decision.

**Rationale.**

- *Same objective in both families.* The stock `Trainer` uses unweighted
  cross-entropy, so without this the transformer and the baseline would optimise
  different things and their macro-F1 would not be comparable — which is the
  whole point of reusing the identical splits (§15).
- *Weighting ≡ perfect oversampling, at no cost.* `w_c = n / (k · n_c)`
  (≈ ×3.3 / ×1.3 / ×0.5 here) multiplies each sample's loss term, giving the
  same expected gradient as duplicating minority rows — but with no extra rows
  on a 400k × 20k sparse matrix and no extra hours on an already multi-hour
  fine-tune.
- *Evaluation stays imbalance-aware.* macro-F1 is the headline (§14) and
  per-class recall is logged, so the effect is visible: NEGATIVE recall 0.732 at
  precision 0.450 — the classic over-predicting signature of a weighted model,
  versus the ~0.3–0.4 recall an unweighted fit gives on a 10% class.

**Alternatives.** Random oversampling — redundant with weighting and slower.
Random undersampling — discards ~230k real rows. **SMOTE** — rejected outright:
interpolating between two sparse 20k-dimensional TF-IDF vectors produces
documents that exist in no language; it is designed for dense, low-dimensional
numeric features.

**Known limitation.** Imbalance is not the binding constraint. NEUTRAL is 25% of
the data and still the worst class (F1 0.479) because the 8.0 cutoff (§3) is
arbitrary — a 7.9 and an 8.1 review are textually indistinguishable. That is
label noise, and no amount of balancing fixes it; richer features (bigrams) or a
different class framing are the levers.

## 18. Engineering-grade run records: comparable, self-describing runs

**Context.** §16 got runs *into* MLflow, but the record was not yet good enough
to answer a reviewer's questions from the UI alone:

- The two trainers logged **different shapes**. The linear baseline logged
  hyperparameters, per-class scores, a confusion matrix, the vectorizer and the
  schema; the transformer logged one parameter (`model_name`) and two metrics.
  Sorting them side by side therefore compared a full row against a stub.
- **No threshold-free metric.** Only F1/accuracy were logged, so a run could not
  be judged on ranking quality (§14 had rejected ROC-AUC outright).
- **The feature store was invisible.** A run pinned the *splits* (`dvc.lock`) but
  said nothing about the ~0.5 GB `feature_store.db` those splits were built
  from, and only one of the two trainers attached the schema.
- **`git_commit` could lie**, since a run from an edited working tree was
  indistinguishable from a clean one.

**Decision.** (2026-08-10)

- *Snapshot, not the store.* `feature_store.snapshot()` returns the store's
  identity and shape — relative path, table, size, mtime, **sha256**, row count,
  column→SQLite-type map and label distribution — and `tracking.start_run` logs
  it as `feature_store/snapshot.json` on **every** run, plus the tag
  `feature_store_sha256` (first 12 chars) so runs can be grouped by input data
  in the UI. The 0.5 GB database itself is never uploaded.
- *Schema logged centrally.* `validation/feature_column.json` moved out of
  `train_linear.py` into `tracking.start_run`, so both families carry the column
  contract they were trained under.
- *ROC-AUC as a secondary metric.* Both trainers log `roc_auc_macro` and
  `roc_auc_weighted` (one-vs-rest). `LinearSVC` has no `predict_proba`, so its
  `decision_function` margins are used — AUC only needs a monotone ranking. The
  transformer softmaxes its logits for the same purpose. A `ValueError` (a
  `--limit` subsample missing a class) is logged and skipped, never fatal.
- *One metric/param vocabulary.* `train_transformer.py` now emits the baseline's
  metric shape (weighted-F1, per-class precision/recall/F1/support, confusion
  matrix artifact) and the same parameter names (`model.*`, `random_state`,
  `limit`, `train_csv`, `test_csv`), and tags smoke runs `smoke_test` like the
  baseline instead of its own `fast_run`.
- *`git_dirty` tag* records whether the working tree was clean.
- *A terminal leaderboard.* `training/compare_runs.py` queries the same store and
  prints one row per run ranked by any metric (`--sort`), excluding smoke runs by
  default, with `--json` for machine consumption.

**Rationale.** A comparison is only as good as its weakest row: identical metric
*names* are what let MLflow (or `compare_runs`) sort a TF-IDF model against a
fine-tuned encoder in one table, and identical parameter names are what make the
difference between two runs readable. Putting the dataset evidence in
`start_run` rather than in each trainer means a new trainer cannot forget it. And
a hash of the feature store is the cheapest possible proof that two runs saw the
same features — a row count alone would not catch a rebuild.

**Alternatives.** Logging `feature_store.db` as an artifact — 0.5 GB per run, and
DVC already versions the file; the hash is the useful part. An MLflow *Dataset*
object (`mlflow.data`) — a natural fit, but it would tie the record to a newer
mlflow API for what is a small JSON document. Per-class ROC-AUC — noisy on the
10% NEGATIVE class and already covered by per-class recall.

<!-- §19 added by Ankita 10/08 -->

## 19. Three training stages, and artifacts named after what they are

**Context.** Two problems surfaced while auditing the run history (2026-08-10):

- The MLflow leaderboard showed only **two** model families. `linear_svc` had no
  run at all: `dvc.yaml` had a single `train` stage running the default
  `logreg`, and the SVC existed only behind the manual `--model linear_svc`
  flag, so `dvc repro` never produced it. `training/metrics_linear_svc.json` on
  disk was the fossil of a one-off run made before tracking existed.
- The transformer artifact was called `distilbert_v1`, but the stage stopped
  fine-tuning DistilBERT — `BASE_MODEL` is `google/bert_uncased_L-4_H-256_A-4`
  (BERT-mini). The directory name asserted weights it did not contain.
- The DVC remote URL in the git-tracked `.dvc/config` was one developer's home
  directory, a path that exists on no other machine.

**Decision.**

- Add a `train_linear_svc` stage to `dvc.yaml`
  (`python -m training.train_linear --model linear_svc`, out
  `model_store/linear_svc_v1.pkl`, metric `training/metrics_linear_svc.json`
  with `cache: false`), with the same deps as `train`.
- Rename `model_store/distilbert_v1{,_checkpoints}` to
  `model_store/bert_mini_v1{,_checkpoints}`, updating `dvc.yaml`,
  `model_store/.gitignore` and `train_transformer.py`.
- Keep only the remote *name* in `.dvc/config`; the URL lives in the git-ignored
  `.dvc/config.local`, set once per machine with
  `dvc remote modify --local`.

**Rationale.** A comparison that can silently lose a model is not a comparison:
a stage is reproducible and shows up in `dvc dag`, a documented CLI flag is a
thing people forget. An artifact name is documentation, and a wrong one is worse
than none — anyone loading `distilbert_v1/` would have drawn the wrong
conclusion about the numbers. And a machine-local absolute path in a shared
config breaks every clone but one, while `config.local` is exactly the layer DVC
provides for per-machine settings.

**Known caveat.** Declaring the stage does not create the run: `linear_svc`
stays absent from the leaderboard until `dvc repro train_linear_svc` is actually
executed. Separately, `compare_runs` filters on `smoke_test`, so the legacy runs
recorded before that tag existed stay hidden unless `--all` is passed.

**Alternatives.** A `foreach` stage over the model names — tidier, but it makes
the two stages share one `outs` template and obscures which metric file belongs
to which model. Deleting the stale `metrics_linear_svc.json` — it is the only
record of that model's score until the stage is run.

<!-- §20 (recurrent baseline) added by Ankita 10/08 -->

## 20. A recurrent baseline as its own stage and its own module

**Context.** The comparison had two poles and nothing in between: bag-of-ngrams
linear models (§13) and a pretrained transformer (§15). The obvious middle term
is a recurrent net trained from scratch — it sees word **order**, which TF-IDF
cannot (§4, §10), but it carries no pretrained knowledge, so it isolates how
much of the transformer's advantage comes from sequence modelling rather than
from pretraining. A first attempt bolted an `rnn_lstm` branch onto
`training/train_linear.py`, which turned out to be the wrong shape: it fed TF-IDF
rows into `pad_sequences` and an `Embedding`, densified the sparse matrix, and
trained on string labels with an integer loss.

**Decision.** (2026-08-10) A separate module `training/train_rnn.py` and a
separate DVC stage `train_rnn` (`cmd: python -m training.train_rnn`, outs
`model_store/rnn_lstm_v1.pt` + `model_store/rnn_lstm_v1_vocab.json`, metric
`training/metrics_rnn_lstm.json` with `cache: false`), so `dvc dag` now shows
`vectorize -> {train, train_linear_svc, train_rnn, train_transformer}`.

- *Its own input representation.* The stage reads the same
  `data/processed/train_v1.csv` / `test_v1.csv` splits but **not** the fitted
  TF-IDF vectorizer, and builds its own word index instead: top
  `NUM_WORDS=20000` tokens, id 0 = padding, id 1 = OOV, sequences right-padded /
  right-truncated to `MAX_LENGTH=200` (right padding is what
  `pack_padded_sequence` expects, so the recurrent layer never reads a pad).
- *Vocabulary fit on the train split only*, exactly as the vectorizer is (§5) —
  a word index built over train+test would leak test vocabulary into the model.
- *Architectures behind `--arch`*: `rnn_lstm` (default), `rnn_bilstm`,
  `rnn_simple`. `nn.Embedding(128, padding_idx=0)` → recurrent layer (128 units)
  → Dropout(0.2) → `nn.Linear(3)`; Adam at 1e-3, weighted cross-entropy,
  3 epochs, batch 128, a 10% validation split, seeded through
  `torch.manual_seed(RANDOM_STATE)`.
- *`class_weight="balanced"`* (sklearn `compute_class_weight`), the same
  cost-sensitive choice as the linear baseline (§17).
- *Tracking parity.* The same `training/tracking.py`, the same
  `review_sentiment` experiment, run name = the architecture, tags
  `stage=train_rnn` / `framework=pytorch` / `smoke_test` plus the automatic
  `git_commit`, `git_dirty` and `feature_store_sha256`, and the §18 metric
  vocabulary (`macro_f1`, `weighted_f1`, `accuracy`, `roc_auc_macro` /
  `roc_auc_weighted`, `per_class.*`). The per-epoch `train.loss` / `val.loss`
  curves go through the new `Run.log_metric_step` helper.

**Rationale.** A branch inside `train_linear.py` cannot work, because the two
families do not share an input: a recurrent net needs an **ordered sequence of
token ids** and TF-IDF is an unordered bag-of-ngrams weighting, so the only way
to reuse the vectorizer is to densify a 20,000-wide sparse matrix over ~400k
training rows — tens of gigabytes for a representation that has already thrown
the word order away. The vocabulary is therefore part of the model, which is why
it is persisted next to the weights as `rnn_lstm_v1_vocab.json`: a `.pt`
checkpoint alone is unusable for inference, exactly as a classifier pickle is unusable
without its vectorizer (§5). Balanced class weights keep the objective identical
to the linear baseline, so the macro-F1 numbers are comparable — unlike the
transformer, which stays on the stock unweighted loss and says so by logging
`model.class_weight="none"`. The consequence is that one leaderboard now spans
three model families — linear, recurrent and transformer — over one set of
splits.

**Framework: PyTorch, not Keras.** The first version of this stage was written
in Keras, and `pip install tensorflow` failed outright with *"Could not find a
version that satisfies the requirement tensorflow (from versions: none)"*: the
project's `.venv` is **Python 3.14**, and TensorFlow publishes no `cp314` wheels
for any platform. Rather than pin the whole project to an older interpreter or
maintain a second one just for this stage, the trainer was ported to `torch`,
which already ships a Python 3.14 build and is already a dependency of
`train_transformer` (§15) — so the recurrent baseline now costs **zero** new
dependencies. The architecture, hyperparameters, metric shape and tracking are
unchanged; only the framework is.

**Known caveat.** As with §19, declaring the stage does not create the run:
`training/metrics_rnn_lstm.json` does not exist and no `rnn_lstm` run appears in
MLflow until `dvc repro train_rnn` is executed. Reverting the ad-hoc branch in
`train_linear.py` also restored the ROC-AUC computation and the
`training/metrics_logreg.json` write it had dropped.

**Alternatives.** A `foreach` stage over the architectures — same objection as
§19, and the three variants would share one `outs` template. Pretrained
embeddings (GloVe/word2vec) for the `Embedding` layer — extra download and
versioning burden, and it blurs the from-scratch/pretrained contrast this stage
exists to draw. Reusing the transformer's tokenizer — a subword vocabulary the
model was never pretrained with, and a hard coupling to the transformer stage.
Keeping Keras behind a second Python 3.12 environment — two interpreters for one
stage, and `dvc repro` would no longer run end to end in one venv.
