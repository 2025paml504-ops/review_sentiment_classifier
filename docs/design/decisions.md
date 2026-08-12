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

## 10. Contraction expansion & negation attachment (v1.1)

**Context.**  Contractions broke into junk tokens and negated praise was lost.

**Decision.** In `features/build_features.py` (added 2026-08-09): Normalize apostrophes, expand contractions, and attach negators to following sentiment words.

**Rationale.** Stdlib‑only, preserves negated sentiment as explicit features.

**Alternatives.** Scope tagging or bigram reliance — heavier and less reliable.

## 11. De-duplicate before cleaning (v1.1)

**Context.** Identical reviews bias classes and waste cleaning work.

**Decision.** Drop duplicates on `[full_review, Reviewer_Score]` immediately
after `combine_reviews()` — i.e. **before** cleaning/tokenization — and
`reset_index(drop=True)`.

**Rationale.** Saves regex passes, prevents leakage, keeps distinct scores.

**Alternatives.** De-duplicating on `clean_review` after cleaning — collapses
slightly more rows (two raw variants can clean to the same string) but costs a
full cleaning pass over the duplicates; not worth it.

## 12. Cleaning defects detected in code (v1.1)

**Context.** Manual samples spotted cleaning issues but missed how often they occurred.

**Decision.** `validation/diagnose_cleaning.py` (added 2026-08-09) scans the
interim CSV for known defects and reports counts/examples. Kept off the DAG,
since a tunable text heuristic shouldn't be able to block a build.

**Rationale.** Turns "does this fix still work" into a count instead of a
guess. Found a real defect the samples missed: 48 placeholder-only reviews
cleaning down to nothing, now dropped in `build_features()`.

## 13. Baseline model — linear classifier on TF-IDF (v1.2)

**Context.** ~504k documents × 20,000 sparse TF-IDF features, three classes at
roughly 10 / 25 / 65 (§3). Need a first model that is cheap, honest to
evaluate, and interpretable.

**Decision.** `training/train_linear.py` trains
`LogisticRegression(solver="saga", class_weight="balanced", max_iter=1000,
tol=1e-3)` as the default, persisting `model_store/logreg_v1.pkl`.
`LinearSVC(class_weight="balanced")` is available via `--model linear_svc`.

**Rationale.** Linear models fit high-dimensional sparse text cheaply (minutes
on CPU) with directly readable per-class coefficients. On the full test split:
LogisticRegression 0.618 macro-F1 / 0.693 accuracy / 0.732 NEGATIVE recall vs.
LinearSVC 0.620 / 0.718 / 0.635 — essentially tied on macro-F1.
LogisticRegression is the default since it recovers more of NEGATIVE (the class
the product cares about) and exposes `predict_proba` for a confidence score.

**Alternatives.** `MultinomialNB` — too weak to ship. Tree ensembles — a poor
fit for 20k sparse features. Transformer fine-tuning — deferred until this
baseline existed (§15).

## 14. Headline metric — macro-F1, not accuracy (v1.2)

**Context.** Classes are ~10 / 25 / 65 imbalanced, so always predicting
POSITIVE already scores ~65% accuracy.

**Decision.** Report **macro-F1** as the headline, alongside per-class
precision/recall/F1 and the confusion matrix; accuracy/weighted-F1 are
recorded but not the target. Git-tracked via a DVC `metrics:` output.

**Rationale.** Macro-F1 weights all three classes equally, so NEGATIVE-class
gains actually move the number. Remaining error is mostly NEUTRAL↔POSITIVE
bleed — expected, since the 8.0 cutoff (§3) is an arbitrary line through a
continuous score.

**Alternatives.** Accuracy — misleading under this imbalance. Weighted-F1 —
inherits the same majority-class bias. ROC-AUC — recorded as a secondary
metric (§18), not the headline.

## 15. Transformer fine-tune as a pipeline stage (v1.2)

**Context.** TF-IDF cannot see word order or context: `not_good` only works
because §10 hand-builds it. A pretrained encoder can, but this was deliberately
deferred until the classical baseline existed (§13).

**Decision.** `training/train_transformer.py` fine-tunes a pretrained encoder
(default `google/bert_uncased_L-4_H-256_A-4`, overridable) via the HuggingFace
`Trainer`, on the **same** splits, writing `model_store/bert_mini_v1/` and its
own metrics JSON. Wired into `dvc.yaml` as its own stage, same shape as `train`.

**Rationale.** Reusing the persisted splits and an identical metric shape keeps
the transformer's macro-F1 directly comparable to the baseline. Kept in the
DAG (reproducible, diffable) despite costing hours and ~2GB of `torch`;
`dvc repro train` stops before it. Saved as a **directory**, not a `.pkl` —
weights, tokenizer and config must travel together, and `safetensors` loads
faster and runs no arbitrary code on load (unlike pickle). BERT-mini over
BERT-base: a fraction of the size/runtime at a modest quality cost, the right
first pass at this corpus size (the artifact used to be named `distilbert_v1`
after an earlier default model — renamed in §19 to match what actually trains).

**Known caveat.** The splits carry `clean_review` (lowercased, negators
glued) — the wrong input distribution for an encoder pretrained on natural
text, understating the transformer's ceiling. Feeding raw `full_review`
instead was tried and reverted (§21).

**Alternatives.** Frozen sentence-transformer embeddings — cheaper, usually
well below a full fine-tune. A hosted LLM API — not open source, no
reproducible artifact to version.

## 16. MLflow for experiment tracking (v1.2)

**Context.** Before this, a training run left behind only a pickle and a
metrics JSON, both overwritten by the next run — nothing recorded which
hyperparameters, code commit, or data version produced a score.

**Decision.** `training/tracking.py` wraps every trainer in an MLflow run,
logging **params** (hyperparameters, `random_state`, split files), **metrics**
(macro-F1, weighted-F1, accuracy, per-class scores), **tags** (`model_type`,
`stage`, `git_commit`, `smoke_test`) and **artifacts** (model, vectorizer,
schema, confusion matrix, a `pip freeze` snapshot, `dvc.lock`). Stored in
`mlflow.db` (SQLite) + `mlruns/`, one shared experiment `review_sentiment`. A
required, declared dependency of every training stage.

**Rationale.** DVC answers "which bytes"; MLflow answers "which knobs, what
score" — bridged by logging `dvc.lock` as the run's dataset reference. SQLite
needs no server and works offline (the legacy `./mlruns` file store is
deprecated as of mlflow 3.15). The store is git-ignored like `model_store/`;
scored metrics stay git-tracked separately via the DVC `metrics:` JSONs. One
shared experiment lets every model family rank against the others.

**Reproducing a run.** Read its `git_commit` tag → check it out → `dvc
checkout` → re-run with the logged parameters. Fixed seeds (`random_state=42`)
make the result identical.

**Alternatives.** W&B / Neptune — hosted, account-gated; this project is
deliberately offline-capable. `dvc exp` — no run-comparison UI or
artifact-per-run store.

## 17. Class imbalance — cost-sensitive loss, not resampling (v1.2)

**Context.** ~10 / 25 / 65 imbalance; always predicting POSITIVE scores ~65%
accuracy while being useless. The stratified split (§5) preserves this for
honest evaluation but does nothing to correct training.

**Decision.** Both model families correct the imbalance in the **loss**, never
the data: `class_weight="balanced"` for LogisticRegression/LinearSVC (§13). The
transformer does **not** yet — it runs stock unweighted cross-entropy, and logs
`model.class_weight="none"` (§18) to make that gap explicit rather than implied.

**Rationale.** Weighting (`w_c = n / (k · n_c)`) is mathematically equivalent to
perfect oversampling, at no cost in extra rows or training time. The effect is
visible in the numbers: NEGATIVE recall 0.732 at precision 0.450 — the classic
over-predicting signature of a weighted model, versus ~0.3–0.4 recall
unweighted.

**Alternatives.** Oversampling — redundant with weighting, slower.
Undersampling — discards ~230k real rows. SMOTE — rejected outright:
interpolating between sparse 20k-dimensional TF-IDF vectors produces documents
in no real language.

**Known limitation.** Imbalance is not the binding constraint — NEUTRAL is 25%
of the data and still the worst class (F1 0.479) because the 8.0 cutoff (§3) is
arbitrary label noise, not an imbalance problem.

## 18. Engineering-grade run records: comparable, self-describing runs (v1.2)

**Context.** §16 got runs into MLflow, but the record was not yet good enough
to compare from the UI alone: the two trainers logged different shapes (one
full row, one a stub), no threshold-free metric existed, the feature store was
invisible to a run, and `git_commit` could lie for a dirty working tree.

**Decision.** `feature_store.snapshot()` (sha256, row count, label
distribution) is logged on every run, plus a `feature_store_sha256` tag; the
schema `feature_column.json` moved into `tracking.start_run` so every trainer
carries it; both trainers log `roc_auc_macro`/`roc_auc_weighted`
(`LinearSVC` uses its `decision_function` margin, having no `predict_proba`);
the transformer now emits the same metric/param vocabulary as the linear
baseline; a `git_dirty` tag records whether the tree was clean;
`training/compare_runs.py` prints a ranked terminal leaderboard.

**Rationale.** A comparison is only as good as its weakest row — identical
metric and parameter names are what let different model families sort against
each other in one table. Putting the dataset evidence in `start_run` rather
than each trainer means a new trainer cannot forget it.

**Alternatives.** Logging the whole feature store as an artifact — 0.5GB per
run; the hash is the useful part. Per-class ROC-AUC — noisy on the 10%
NEGATIVE class, already covered by per-class recall.

## 19. Three training stages, and artifacts named after what they are (v1.2)

**Context.** Auditing the run history found two problems: `linear_svc` had no
DVC stage (only a manual `--model` flag), so it never appeared via `dvc
repro`; the transformer's artifact was still named `distilbert_v1` though the
stage actually fine-tunes BERT-mini. The DVC remote URL was also a
machine-local path committed to shared config.

**Decision.** Added a `train_linear_svc` stage to `dvc.yaml`. Renamed the
transformer artifacts `distilbert_v1` → `bert_mini_v1`. Moved the remote URL
out of the git-tracked `.dvc/config` into the git-ignored `.dvc/config.local`.

**Rationale.** A comparison that can silently lose a model is not a comparison
— a declared stage always shows up in `dvc dag`; a manual flag is easy to
forget. A wrong artifact name is worse than none. A machine-local path in
shared config breaks every other clone.

**Alternatives.** A `foreach` stage over model names — obscures which metric
file belongs to which model.

## 20. A recurrent baseline as its own stage and its own module (v1.2)

**Context.** The comparison had two poles — bag-of-ngrams linear models (§13)
and a pretrained transformer (§15) — with nothing in between. A recurrent net
trained from scratch sees word order, unlike TF-IDF, but carries no pretrained
knowledge, isolating how much of the transformer's edge comes from sequence
modelling versus pretraining. An early attempt bolted an RNN branch onto
`train_linear.py`, which was the wrong shape — it densified TF-IDF into an
`Embedding`.

**Decision.** A separate module `training/train_rnn.py` and DVC stage
`train_rnn`, writing `model_store/rnn_lstm_v1.pt` + a vocab JSON + its own
metrics JSON. Builds its own word index from the train split (not TF-IDF):
`NUM_WORDS=20000`, sequences padded/truncated to 200. Architectures behind
`--arch`: `rnn_lstm` (default), `rnn_bilstm`, `rnn_simple` — Embedding →
recurrent layer (128 units) → Dropout → Linear(3); Adam, weighted
cross-entropy, 3 epochs. Same `class_weight="balanced"` treatment and tracking
parity as the linear baseline.

**Rationale.** A recurrent net needs an ordered sequence of token ids; TF-IDF
is an unordered bag-of-ngrams, so reusing the vectorizer would mean densifying
a 20,000-wide sparse matrix over ~400k rows — tens of gigabytes, defeating the
point. The vocabulary is persisted alongside the weights since a `.pt`
checkpoint alone is unusable without it, same as a classifier pickle without
its vectorizer.

**Framework: PyTorch, not Keras.** The first version was Keras; `pip install
tensorflow` failed outright since the project's Python 3.14 venv has no
TensorFlow wheels. Ported to `torch`, already a dependency of the transformer
stage — zero new dependencies.

**Alternatives.** A `foreach` stage over architectures — same objection as
§19. Pretrained embeddings (GloVe/word2vec) — blurs the from-scratch/pretrained
contrast this stage exists to draw. Keeping Keras behind a second Python
environment — breaks the one-venv `dvc repro`.

## 21. Tuning experiments — kept vs. reverted (v1.2 / v1.3 — see the Version column)

**Context.** Every knob below was changed, measured against the standing
metrics, and judged on macro-F1 (§14) before a decision was made either way.
Nothing here is a guess: each row is a real run recorded in MLflow, comparable
by `git_commit` (§16). Listing the ones that *didn't* help is as much a part
of the record as the ones that did — a config that was tried and dropped is
easy to accidentally re-discover a second time without this table.

| Experiment | Change | Result | Decision | Version |
|---|---|---|---|---|
| TF-IDF vocabulary width | `max_features` 20,000 → 30,000 with trigrams `(1,3)` | `saga` failed to converge; macro-F1 fell to 0.537 | Reverted | v1.2 |
| TF-IDF vocabulary width (retry) | `max_features` 20,000 → 24,000, bigrams `(1,2)` | Still non-convergent; macro-F1 0.556 | Reverted to the original 20,000 / `(1,2)` | v1.2 |
| RNN epochs | 3 → 4 | val_loss stopped improving after epoch 3; macro-F1 fell 0.6436 → 0.6393 | Reverted to 3 | v1.2 |
| RNN epochs (re-verified on the post-bugfix, larger dataset) | 3 vs. 4 vs. 6 | 3 → 0.6488, 4 → 0.6334, 6 → 0.6438 — 3 still wins, and the dip-then-partial-recovery pattern reproduced identically (same loss values) across separate runs, confirming training is deterministic | Kept at 3 | v1.3 |
| Sentiment thresholds | Scheme A (`<6`/`6–8`/`≥8`) → NPS-style (`<7`/`7–9`/`≥9`) | Macro-F1 improved on every model | Reverted — prioritizes accuracy over macro-F1 as the headline metric, a deliberate call, not a measurement failure | v1.2 |
| Transformer input text | `clean_review` → raw `full_review` | Macro-F1 0.6461 → 0.6459 — a wash, likely masked by `MAX_LENGTH=64` truncating before the difference could show | Reverted to `clean_review` | v1.2 |

**Rationale.** The alternative to a table like this is tribal knowledge — "we
tried that, it didn't work" with no record of what "that" was or what "didn't
work" measured. Every row here is reproducible from its `git_commit` tag (§16)
even though the code itself has since moved past most of these configs.

**How to read a "reverted" row.** Reverted means the code went back to its
prior value — it does not mean the experiment was wasted. Each one is a
negative result worth having on record, in the same spirit as §12's cleaning
diagnostics: a documented dead end is what stops the next person (including a
future run of this same project) from re-spending the time to re-discover it.

## 22. Serving `logreg`, not the highest-scoring model (v1.4)

**Context.** M4 requires a REST API serving one trained model. Four exist;
`bert_mini` has the best macro-F1 (0.6685 historically; 0.6461 on current
data), but every trained model is a legitimate candidate.

**Decision.** `serving/app.py` (FastAPI) serves `logreg`, loading
`model_store/logreg_v1.pkl` and the fitted vectorizer once at startup, not
per-request. Input is validated in three layers: Pydantic's `min_length=1`
rejects an empty string, a custom validator rejects whitespace-only text, and
a post-`clean_text()` check rejects text that cleans down to nothing
(punctuation/numbers/placeholder content) — the same "empty document" case
`build_features()` already drops at training time (§1). Malformed JSON, a
missing field, or a wrong type are caught by FastAPI's built-in validation.
Every response includes its own `latency_ms`.

**Rationale.** Serving latency is a different constraint than training-time
macro-F1. `logreg` loads from a small pickle in milliseconds and needs only
`pandas`/`scikit-learn`/`joblib` at runtime; `bert_mini` needs `torch` and
`transformers` resident in memory just to answer one request — multiple
seconds of cold-start weight, and a ~2GB dependency footprint, for a macro-F1
gain that's marginal on current data (0.6461 vs. `logreg`'s 0.6252) and not
worth the latency cost for a first API. Measured locally: ~336 req/s,
~3ms/request average, sequential, no batching — see [serving/README.md](../../serving/README.md).
Loading the model once at import time, not per-request, is what makes that
number meaningful; reloading a pickle on every call would dominate latency
far more than inference itself does.

**Alternatives.** Serving `bert_mini` — rejected for the latency/dependency
cost above; revisit once there's an actual accuracy requirement that
justifies it. A `--model` query param to pick any of the four at request time
— more flexible, but multiplies the validation/testing surface for a first
version, and mixing model families behind one endpoint blurs which one a
caller actually got. Silently returning a low-confidence prediction on blank
input instead of a `422` — rejected: a prediction on no real content is worse
than an honest error, and it would go undetected by whoever's calling the API.
