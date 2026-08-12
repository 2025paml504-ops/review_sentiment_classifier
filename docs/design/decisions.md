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

**Context.** Contractions were breaking into junk tokens, and negated praise
was getting lost.

**Decision.** In `features/build_features.py`: normalize apostrophes, expand
contractions, then attach negators to the word right after them.

**Rationale.** No new dependencies, and negation stays visible as a real
feature instead of disappearing.

**Alternatives.** Scope tagging or just relying on bigrams - both heavier and
less reliable.

## 11. De-duplicate before cleaning (v1.1)

**Context.** Duplicate reviews skew the classes and waste cleaning time.

**Decision.** Drop duplicates on `[full_review, Reviewer_Score]` right after
`combine_reviews()`, before any cleaning happens, then reset the index.

**Rationale.** Fewer regex passes, no leakage, and distinct scores are kept.

**Alternatives.** Could dedupe on `clean_review` after cleaning instead -
catches a few more near-duplicates, but means running the full cleaning pass
over rows that get thrown away anyway. Not worth it.

## 12. Cleaning defects detected in code (v1.1)

**Context.** Manual spot-checks had found cleaning issues, but there was no
way to tell how often they actually occurred.

**Decision.** `validation/diagnose_cleaning.py` scans the interim CSV for
known defects and reports counts and examples. It stays off the DAG - a text
heuristic shouldn't be able to block a build.

**Rationale.** Turns "is this fix still working" into a number instead of a
guess. It actually found something the manual checks missed: 48 reviews that
were only placeholder text and cleaned down to nothing. Those get dropped in
`build_features()` now.

## 13. Baseline model — linear classifier on TF-IDF (v1.2)

**Context.** ~504k documents, 20,000 sparse TF-IDF features, three classes
split roughly 10/25/65 (§3). Needed a first model that's cheap, honest to
evaluate, and easy to reason about.

**Decision.** `training/train_linear.py` trains a
`LogisticRegression(solver="saga", class_weight="balanced", max_iter=1000,
tol=1e-3)` by default and saves it to `model_store/logreg_v1.pkl`.
`LinearSVC(class_weight="balanced")` is available too, via `--model linear_svc`.

**Rationale.** Linear models handle high-dimensional sparse text well and
train in minutes on CPU, and the per-class coefficients are actually readable.
On the full test split, LogisticRegression scored 0.618 macro-F1 / 0.693
accuracy / 0.732 NEGATIVE recall; LinearSVC scored 0.620 / 0.718 / 0.635 -
basically tied on macro-F1. LogisticRegression won out as the default because
it catches more of the NEGATIVE class, which is what the product actually
cares about, and it gives a real confidence score through `predict_proba`.

**Alternatives.** `MultinomialNB` is a fine sanity check but too weak to ship.
Tree ensembles don't suit 20k sparse features well. Transformer fine-tuning
came later, once this baseline existed (§15).

## 14. Headline metric — macro-F1, not accuracy (v1.2)

**Context.** The classes are imbalanced, roughly 10/25/65, so a model that
always guesses POSITIVE already gets ~65% accuracy without learning anything.

**Decision.** Report macro-F1 as the headline number, along with per-class
precision/recall/F1 and the full confusion matrix. Accuracy and weighted-F1
are still recorded, just not the target. This goes out as a DVC `metrics:`
output, so it's git-tracked.

**Rationale.** Macro-F1 treats all three classes equally, so a gain on the
small NEGATIVE class actually shows up in the number. Most of what's left
over is NEUTRAL and POSITIVE bleeding into each other, which makes sense
given the 8.0 cutoff from §3 is just an arbitrary line through a continuous
score.

**Alternatives.** Accuracy is misleading here given the imbalance.
Weighted-F1 has the same majority-class bias baked in. ROC-AUC is tracked as
a secondary metric (§18) but isn't the headline.

## 15. Transformer fine-tune as a pipeline stage (v1.2)

**Context.** TF-IDF can't see word order or context - `not_good` only works
because §10 builds it by hand. A pretrained encoder actually understands
this, but it was deliberately put off until the classical baseline existed
(§13).

**Decision.** `training/train_transformer.py` fine-tunes a pretrained encoder
(`google/bert_uncased_L-4_H-256_A-4` by default, overridable) through
HuggingFace's `Trainer`, on the same splits as everything else, writing out
`model_store/bert_mini_v1/` and its own metrics file. It's its own stage in
`dvc.yaml`, shaped the same way as `train`.

**Rationale.** Using the same splits and the same metric shape means the
transformer's macro-F1 is directly comparable to the baseline's, no caveats
needed. It stays in the DAG even though it costs hours and ~2GB of `torch` -
`dvc repro train` stops before reaching it if you don't need it. It's saved
as a directory, not a `.pkl`, because the weights, tokenizer, and config all
have to travel together, and `safetensors` both loads faster and doesn't
execute arbitrary code the way pickle can. BERT-mini over BERT-base because
it's a fraction of the size and runtime for a modest hit in quality - the
right first pass at this size of corpus. (The artifact used to be called
`distilbert_v1`, left over from an earlier default model - renamed in §19 to
match what's actually being trained.)

**Known caveat.** The splits carry `clean_review`, which is lowercased,
stripped, and has negators glued together. That's not really the kind of
text an encoder pretrained on natural writing expects, so it probably
understates what the transformer could actually do. Feeding it raw
`full_review` instead was tried and reverted (§21).

**Alternatives.** Frozen sentence-transformer embeddings are cheaper but
usually land well below a full fine-tune. A hosted LLM API was ruled out -
not open source, per-call cost, and nothing reproducible to version.

## 16. MLflow for experiment tracking (v1.2)

**Context.** Before this, a training run left behind just a pickle and a
metrics file, and both got overwritten the next time you trained. Nothing
recorded which hyperparameters produced a given score, which commit the code
was at, or which version of the data was used.

**Decision.** `training/tracking.py` wraps every trainer in an MLflow run.
Each run logs its parameters (hyperparameters, `random_state`, which split
files), its metrics (macro-F1, weighted-F1, accuracy, per-class scores), tags
(`model_type`, `stage`, `git_commit`, `smoke_test`), and artifacts (the
model, the vectorizer, the schema, the confusion matrix, a `pip freeze`
snapshot, `dvc.lock`). Everything goes into a local SQLite store, `mlflow.db`,
plus `mlruns/`, under one shared experiment called `review_sentiment`. It's a
required dependency now, not optional - every training stage declares it.

**Rationale.** DVC tells you which bytes went in and came out; MLflow tells
you which knobs were turned and what score came out. `dvc.lock` is the
bridge between them - it gets logged as the run's dataset reference. SQLite
means no server to run, and it works offline (the old `./mlruns` file store
is actually deprecated as of mlflow 3.15 anyway). The store itself is
git-ignored, same rule as `model_store/`; the scored metrics stay git-tracked
separately through the DVC `metrics:` files. Putting every model family in
one experiment means they can all be ranked against each other.

**Reproducing a run.** Grab its `git_commit` tag, check that commit out,
`dvc checkout`, then re-run with the logged parameters. Seeds are fixed
(`random_state=42` everywhere), so you get the same result back.

**Alternatives.** W&B and Neptune are hosted and need an account - this
project is meant to run fully offline and open source. `dvc exp` is good for
parameter sweeps tied to the DAG but doesn't give a comparison UI or a
per-run artifact store.

## 17. Class imbalance — cost-sensitive loss, not resampling (v1.2)

**Context.** The imbalance is roughly 10% NEGATIVE, 25% NEUTRAL, 65%
POSITIVE. Always guessing POSITIVE gets ~65% accuracy while being useless,
and NEGATIVE - the class that actually matters for the business - is the one
an unweighted model is most likely to ignore. The split is stratified (§5),
which keeps that imbalance intact for honest evaluation but doesn't help
training at all.

**Decision.** Both linear models fix the imbalance in the loss, not the
data - `LogisticRegression` and `LinearSVC` both use `class_weight="balanced"`
(§13). The transformer doesn't do this yet: it's still running plain
unweighted cross-entropy, and now logs `model.class_weight="none"` (§18) so
that gap is visible instead of just assumed.

**Rationale.** Weighting each class by `w_c = n / (k · n_c)` has roughly the
same effect as perfectly oversampling the minority classes, without adding
rows or training time. The effect shows up directly in the numbers: NEGATIVE
recall sits at 0.732 with precision at 0.450, which is exactly the
over-predicting pattern you'd expect from a weighted model - an unweighted
one gets more like 0.3-0.4 recall on a class this small.

**Alternatives.** Oversampling does roughly the same thing as weighting but
slower. Undersampling would throw away ~230k real rows. SMOTE was ruled out
completely - interpolating between two sparse 20,000-dimensional TF-IDF
vectors just produces documents that aren't real language; it's built for
dense, low-dimensional features.

**Known limitation.** Imbalance isn't actually the bottleneck here. NEUTRAL
is 25% of the data and is still the worst-performing class (F1 0.479),
because the 8.0 cutoff from §3 is arbitrary - a review scored 7.9 and one
scored 8.1 read almost identically. That's label noise, not an imbalance
problem, and no amount of reweighting fixes it.

## 18. Engineering-grade run records: comparable, self-describing runs (v1.2)

**Context.** §16 got runs into MLflow, but the record wasn't good enough yet
to answer questions from the UI alone. The two trainers logged completely
different shapes - the linear baseline logged hyperparameters, per-class
scores, a confusion matrix, the vectorizer, and the schema; the transformer
logged one parameter and two metrics. Comparing them side by side meant
comparing a full row against basically a stub. There was also no
threshold-free metric, the feature store itself was invisible to any given
run, and a `git_commit` tag could lie if the working tree had uncommitted
changes.

**Decision.** `feature_store.snapshot()` now returns a sha256, row count, and
label distribution for the store, and every run logs it, plus a short
`feature_store_sha256` tag. The schema file, `feature_column.json`, moved
into `tracking.start_run` so both trainers carry it automatically. Both
trainers now log `roc_auc_macro` and `roc_auc_weighted` too - `LinearSVC`
doesn't have `predict_proba`, so it uses its `decision_function` margin
instead, which works fine since AUC just needs a ranking. The transformer
now emits the same metric and parameter names as the linear baseline. A
`git_dirty` tag records whether the tree was actually clean. And
`training/compare_runs.py` prints a ranked leaderboard right in the
terminal.

**Rationale.** A comparison is only as useful as its weakest row. Having the
same metric and parameter names across model families is what lets a TF-IDF
model and a fine-tuned encoder sort into the same table. Putting the dataset
evidence into `start_run` instead of each trainer means a future trainer
can't forget to log it.

**Alternatives.** Logging the whole feature store as an artifact would mean
0.5GB per run for no real benefit - the hash already proves what matters.
Per-class ROC-AUC turned out too noisy on the 10% NEGATIVE class, and
per-class recall already covers that ground.

## 19. Three training stages, and artifacts named after what they are (v1.2)

**Context.** Auditing the run history turned up two problems. `linear_svc`
had no DVC stage at all - `dvc.yaml` only had a `train` stage running the
default `logreg`, and `linear_svc` only existed behind a manual `--model`
flag, so it never showed up from a plain `dvc repro`. The metrics file for it
on disk was just a leftover from a one-off run made before tracking even
existed. Separately, the transformer's artifact was still named
`distilbert_v1`, even though the stage had moved on to fine-tuning BERT-mini
- the folder name was asserting weights it didn't actually contain. On top
of that, the DVC remote URL committed in `.dvc/config` was one developer's
home directory, which obviously doesn't exist on anyone else's machine.

**Decision.** Added a real `train_linear_svc` stage to `dvc.yaml`. Renamed
the transformer artifacts from `distilbert_v1` to `bert_mini_v1`. Moved the
remote URL out of the git-tracked `.dvc/config` and into the git-ignored
`.dvc/config.local`.

**Rationale.** A comparison that can silently drop a model isn't really a
comparison - a declared stage always shows up in `dvc dag`, but a flag people
have to remember is easy to forget. A wrong artifact name is worse than no
name at all - someone loading `distilbert_v1/` would have drawn the wrong
conclusion entirely. And a machine-local path in a shared config file breaks
the project for every other clone.

**Alternatives.** A `foreach` stage over the model names would be tidier, but
it obscures which metric file actually belongs to which model.

## 20. A recurrent baseline as its own stage and its own module (v1.2)

**Context.** The comparison had two extremes and nothing in between -
bag-of-ngrams linear models (§13) on one side, a pretrained transformer
(§15) on the other. The natural middle ground is a recurrent net trained
entirely from scratch: it can see word order, which TF-IDF can't, but it has
no pretrained knowledge going in, which makes it useful for separating how
much of the transformer's advantage comes from sequence modeling versus just
pretraining. An early attempt tried bolting an RNN branch onto
`train_linear.py`, but that was the wrong shape - it meant feeding TF-IDF
rows into `pad_sequences` and an `Embedding` layer, which doesn't make sense.

**Decision.** A separate module, `training/train_rnn.py`, with its own DVC
stage `train_rnn`, writing out `model_store/rnn_lstm_v1.pt`, a vocab JSON,
and its own metrics file. It reads the same splits as everything else but
builds its own word index instead of using TF-IDF: top 20,000 words,
sequences padded or truncated to 200 tokens. Architecture is chosen with
`--arch` - `rnn_lstm` by default, plus `rnn_bilstm` and `rnn_simple` -
Embedding into a 128-unit recurrent layer, Dropout, then a Linear(3) head.
Trained with Adam, weighted cross-entropy, 3 epochs. Same
`class_weight="balanced"` treatment and the same tracking setup as the linear
baseline.

**Rationale.** A recurrent net needs an ordered sequence of token ids, and
TF-IDF is an unordered bag of n-grams - reusing the vectorizer would mean
densifying a 20,000-wide sparse matrix across ~400k rows, which is tens of
gigabytes for no good reason. The vocabulary gets saved right alongside the
weights because a `.pt` checkpoint on its own is useless without it, the same
idea as a classifier pickle needing its vectorizer.

**Framework: PyTorch, not Keras.** The first version of this was actually
written in Keras, but `pip install tensorflow` failed outright - the project
runs on Python 3.14, and TensorFlow doesn't publish wheels for it. Ported to
`torch` instead, which was already a dependency for the transformer stage, so
this added zero new dependencies. The architecture, hyperparameters, and
tracking are all unchanged - only the framework is different.

**Alternatives.** A `foreach` stage over the architectures runs into the same
problem as §19. Pretrained embeddings like GloVe or word2vec would blur the
from-scratch-versus-pretrained comparison this stage exists to draw. Keeping
Keras around in a second Python environment would mean two interpreters for
one stage, and `dvc repro` wouldn't run end to end anymore.

## 21. Tuning experiments — kept vs. reverted (v1.2 / v1.3 — see the Version column)

**Context.** Every change listed below was actually made, measured against
the current numbers, and judged on macro-F1 (§14) before deciding to keep it
or roll it back. None of this is guesswork - each row is a real run sitting
in MLflow, traceable through its `git_commit` tag (§16). The ones that didn't
help are worth recording just as much as the ones that did, since a config
that's already been tried and dropped is easy to accidentally try again
without a record like this.

| Experiment | Change | Result | Decision | Version |
|---|---|---|---|---|
| TF-IDF vocabulary width | `max_features` 20,000 → 30,000 with trigrams `(1,3)` | `saga` failed to converge; macro-F1 fell to 0.537 | Reverted | v1.2 |
| TF-IDF vocabulary width (retry) | `max_features` 20,000 → 24,000, bigrams `(1,2)` | Still non-convergent; macro-F1 0.556 | Reverted to the original 20,000 / `(1,2)` | v1.2 |
| RNN epochs | 3 → 4 | val_loss stopped improving after epoch 3; macro-F1 fell 0.6436 → 0.6393 | Reverted to 3 | v1.2 |
| RNN epochs (re-verified on the post-bugfix, larger dataset) | 3 vs. 4 vs. 6 | 3 → 0.6488, 4 → 0.6334, 6 → 0.6438 — 3 still wins, and the dip-then-partial-recovery pattern reproduced identically (same loss values) across separate runs, confirming training is deterministic | Kept at 3 | v1.3 |
| Sentiment thresholds | Scheme A (`<6`/`6–8`/`≥8`) → NPS-style (`<7`/`7–9`/`≥9`) | Macro-F1 improved on every model | Reverted — prioritizes accuracy over macro-F1 as the headline metric, a deliberate call, not a measurement failure | v1.2 |
| Transformer input text | `clean_review` → raw `full_review` | Macro-F1 0.6461 → 0.6459 — a wash, likely masked by `MAX_LENGTH=64` truncating before the difference could show | Reverted to `clean_review` | v1.2 |

**Rationale.** Without a table like this, all you have is tribal knowledge -
"we tried that, it didn't work" with no record of what "that" actually was or
what "didn't work" was measured against. Every row here can be reproduced
from its `git_commit` tag (§16), even though the code has moved on from most
of these configs by now.

**How to read a "reverted" row.** Reverted just means the code went back to
what it was before - it doesn't mean the experiment was wasted. Each one is a
real negative result worth keeping on record, the same idea as §12's cleaning
diagnostics: writing down a dead end is what stops someone (including a
future version of this same project) from spending time rediscovering it.

## 22. Serving `logreg`, not the highest-scoring model (v1.4)

**Context.** M4 needs a REST API serving one trained model. There are four
to choose from - `bert_mini` has the best macro-F1 (0.6685 historically,
0.6461 on the current data) - but any of the four could reasonably serve.

**Decision.** `serving/app.py`, built with FastAPI, serves `logreg`. It loads
`model_store/logreg_v1.pkl` and the fitted vectorizer once at startup, not on
every request. Input goes through three layers of validation: Pydantic's
`min_length=1` catches an empty string outright, a custom validator catches
whitespace-only text, and a check after `clean_text()` catches text that
cleans down to nothing - punctuation, numbers, placeholder content - which is
the same "empty document" case `build_features()` already drops at training
time (§1). Malformed JSON, a missing field, or the wrong type are all caught
automatically by FastAPI's own validation. Every response comes back with its
own `latency_ms`.

**Rationale.** Serving has a different constraint than training-time
macro-F1: latency. `logreg` loads from a small pickle in milliseconds and
only needs `pandas`, `scikit-learn`, and `joblib` at runtime. `bert_mini`
needs `torch` and `transformers` sitting in memory just to answer a single
request - several seconds of cold-start time and a ~2GB dependency footprint,
for a macro-F1 gain that's actually pretty marginal on the current data
(0.6461 versus `logreg`'s 0.6252). Not worth that cost for a first version of
the API. Measured locally: about 336 requests/second, roughly 3ms per request
on average, sequential, no batching (see
[serving/README.md](../../serving/README.md)). Loading the model once at
startup instead of per-request is what makes that number mean anything -
reloading a pickle on every call would swamp the actual inference time.

**Alternatives.** Serving `bert_mini` instead was ruled out for the latency
and dependency reasons above - worth revisiting if there's ever an actual
accuracy requirement that justifies it. A `--model` query param to pick any
of the four at request time would be more flexible, but it multiplies how
much needs testing for a first version, and mixing model families behind one
endpoint makes it unclear which one a caller actually got. Silently returning
a low-confidence prediction on blank input instead of a `422` was also
considered and rejected - a prediction with nothing real behind it is worse
than an honest error, and it would go unnoticed by whoever's calling the
API.
