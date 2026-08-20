# Decision-making guide

[← Design docs](README.md) · Related: [Architecture](architecture.md) · [Pipeline](../pipeline.md) · [Versioning](../versioning.md)

Lightweight ADR-style records of the key choices behind the pipeline. Each entry
gives the **context**, the **decision**, the **rationale**, and the
**alternatives** considered, so future changes can be made deliberately.

## At a glance

| # | Decision | Why |
|---|---|---|
| [1](#1-text-cleaning--tokenization) | Clean text with plain regex: lowercase, expand contractions, strip placeholder text, attach negation to the next word | No extra dependencies; keeps negation ("not good") from being lost |
| [2](#2-combine-positive--negative-into-full_review) | Merge the positive and negative review fields into one document | One sentiment label per review needs one text field |
| [3](#3-sentiment-labeling--vader-on-review-text-binary-v15) | Label sentiment with VADER on the review text itself, binary NEGATIVE/POSITIVE | The old score-based labels sometimes disagreed with what the review actually said; text-derived labels don't |
| [4](#4-tf-idf-over-embeddings) | Use TF-IDF features, not embeddings, for the first model | Fast, cheap, and easy to explain - the right starting point |
| [5](#5-fit-on-train-only--stratified-split) | Split the data before fitting TF-IDF, and fit on the train split only | Prevents test-data leakage into the features |
| [6](#6-feature-store--sqlite) | Store features in SQLite, not MySQL | No server needed; zero setup |
| [7](#7-interim-as-csv-tokens-as-json-string) | Keep the intermediate file as CSV, with tokens stored as JSON text | Avoids adding a Parquet dependency just for one list column |
| [8](#8-schema-contract-as-json) | Define the expected columns/types in a JSON file, not in code | A schema change becomes a one-line edit |
| [9](#9-versioning-with-dvc) | Version datasets and model files with DVC | They're too large for git; a commit hash still pins an exact version |
| [10](#10-contraction-expansion--negation-attachment-v11) | Expand contractions, then attach negation words to the next word | Keeps "not good" from splitting into two unrelated, unhelpful tokens |
| [11](#11-de-duplicate-before-cleaning-v11) | Remove duplicate reviews before cleaning | Saves time and stops duplicates from skewing the classes |
| [12](#12-cleaning-defects-detected-in-code-v11) | Added a script that scans for known cleaning bugs and reports counts | Turns "is the fix still working?" into a number, not a guess |
| [13](#13-baseline-model--linear-classifier-on-tf-idf-v12-retrained-v15) | Default model is a LogisticRegression (`class_weight="balanced"`); a calibrated LinearSVC is the alternative | `logreg` macro-F1 0.8358, `linear_svc` 0.8652 on the current binary data |
| [14](#14-headline-metric--macro-f1-not-accuracy-v12) | Report macro-F1, not accuracy, as the main score | Accuracy can look good while ignoring the smallest class; macro-F1 can't |
| [15](#15-transformer-fine-tune-as-a-pipeline-stage-v12-retrained-v15) | Fine-tune a small pretrained model (now BERT-tiny, v1.5) as a fourth model | distilbert-base-uncased was tried first but too slow on CPU (~20h/epoch) |
| [16](#16-mlflow-for-experiment-tracking-v12) | Log every run's settings, scores, and library versions to MLflow | Makes any past run reproducible from its own record |
| [17](#17-class-imbalance--cost-sensitive-loss-not-resampling-v12) | Weight the rare class more heavily during training, for all 4 models | NEGATIVE recall across the four ranges 0.70-0.93, `logreg` included |
| [18](#18-engineering-grade-run-records-v12) | Standardize what every run logs, so runs can be compared fairly | A comparison is only as good as its weakest, most incomplete entry |
| [19](#19-three-training-stages-correctly-named-v12) | Gave every model its own pipeline stage and an accurate artifact name | A model behind a manual flag, or a wrongly named folder, is easy to miss or misread |
| [20](#20-a-recurrent-model-as-a-third-baseline-v12) | Added an RNN/LSTM trained from scratch as a third model | Sits between the simple baseline and the transformer - learns word order, no pretraining |
| [21](#21-tuning-experiments--kept-vs-reverted-v12--v13--v14--v15) | Kept a running log of every parameter change tried, and whether it was kept | Stops the same dead end from being tried again by accident |
| [22](#22-serving-rnn_lstm-the-highest-scoring-model-on-current-data-v14-re-confirmed-v15) | The API serves `rnn_lstm`, the model with the best macro-F1 (re-confirmed after full retrain, v1.5) | Wins by a clear margin (0.8918 vs. next-best 0.8652) on the metric this project has used throughout |
| [23](#23-a-plain-html-page-over-the-api-not-a-framework-app-v14) | A single plain HTML/CSS/JS page (`ui/index.html`), no framework, calls the API | The whole page is one form and one API call - too simple to need framework overhead |

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

## 3. Sentiment labeling — VADER on review text, binary (v1.5)

**Context.** The dataset has a numeric `Reviewer_Score` (2.5–10.0) but no
explicit sentiment label. This project used that score for labeling
through v1.4 (a 3-class Scheme A: `NEGATIVE < 6`, `6 ≤ NEUTRAL < 8`,
`POSITIVE ≥ 8`, ~10/25/65 split). Manually checking reviews near that
boundary found real cases where the score and the review text disagreed -
a review reading "staff rude unhelpful money grabbers" still scored 8.8/10.

**Decision.** Sentiment now comes from **VADER** (a lexicon-based sentiment
scorer) run directly on `full_review`, not from `Reviewer_Score`. Binary,
not 3-class: `compound >= 0.0` -> POSITIVE, else NEGATIVE
(`features/build_features.py`, `vaderSentiment` package). Produces a
86.5% / 13.5% POSITIVE/NEGATIVE split (435,283 / 68,163 rows). The old
Scheme A code is kept, commented out, not deleted.

**Rationale.** Labeling from the review text itself, not a separate
numeric rating, removes the score-vs-text disagreement case above by
construction - the label now describes what the text actually says. The
`0.0` cutoff is VADER's own built-in zero-point (net-positive vs
net-negative lexicon weight), not a number picked to hit a target balance.
A mathematically closer-to-50/50 cutoff (0.70) was tried and rejected for
the same reason Scheme A's alternatives were: manually checking reviews
near it found genuinely positive text ("very helpful polite staff") that
cutoff would have called NEGATIVE.

**Alternatives.** Reassigning `Reviewer_Score`'s old NEUTRAL band into a
binary split via a new score cutoff was considered first; VADER was chosen
instead specifically because it addresses the score-vs-text disagreement
problem that motivated this change in the first place, which a different
score cutoff alone would not. NLTK's own copy of VADER was considered
over the standalone `vaderSentiment` package - not used, since NLTK's
version needs a separate `nltk.download('vader_lexicon')` call at runtime,
a reproducibility risk this project avoids elsewhere (§16). This is a
genuine labeling-contract change - see [Versioning](../versioning.md)'s
naming convention for why it's treated differently from the earlier
Scheme A tuning experiments (§21) that didn't change the contract.

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

## 13. Baseline model — linear classifier on TF-IDF (v1.2, retrained v1.5)

**Context.** Needed a first, simple model to measure everything else against.

**Decision.** `training/train_linear.py` trains a LogisticRegression by
default (`model_store/logreg_v1.pkl`), with `class_weight="balanced"`. A
calibrated LinearSVC is also available via `--model linear_svc`. Briefly
changed (v1.5): `logreg` dropped `class_weight="balanced"` at the user's
explicit request, matching another branch's unweighted config; measuring
the effect (§17, §21) showed it cost NEGATIVE recall without a compensating
gain, so it was restored to weighted later the same version.

**Rationale.** On the current binary, VADER-labeled data: `logreg` scores
macro-F1 0.8358 / accuracy 0.9073; `linear_svc` (calibrated) scores 0.8652 /
0.9412. Both are far higher than any 3-class result this project measured
before - binary classification is a simpler task, and labels derived from
the review text itself correlate more directly with a text-based model's
own features than a separate numeric score did (§3). `logreg` scores lower
than `linear_svc` on macro-F1 here specifically because of the weighting
tradeoff measured in §17 - it isn't a weaker model overall.

**Alternatives.** Naive Bayes was too weak to use. Tree models don't suit
this much sparse text. The transformer came later, once this baseline
existed (§15).

## 14. Headline metric — macro-F1, not accuracy (v1.2)

**Context.** The classes are imbalanced - originally ~10/25/65 across three
classes; now (v1.5, binary VADER labels, §3) ~13.5/86.5 across two - so
accuracy alone can look good while mostly ignoring the small class either way.

**Decision.** Macro-F1 (an equal-weighted average across classes) is the
main score. Accuracy and weighted-F1 are still recorded, just not the
target.

**Rationale.** Macro-F1 can't be inflated by doing well only on the largest
class - a gain on the small NEGATIVE class actually shows up in the number.
This kept mattering after the move to binary labels: `rnn_lstm` pulls ahead
of all three other models specifically on NEGATIVE-class handling (§22).

**Alternatives.** Accuracy and weighted-F1 both hide poor performance on the
small class. ROC-AUC is tracked too (§18) but isn't the headline.

## 15. Transformer fine-tune as a pipeline stage (v1.2, retrained v1.5)

**Context.** TF-IDF can't see word order or context. A pretrained language
model can - this was built after the simpler baseline existed (§13), not
instead of it.

**Decision.** `training/train_transformer.py` fine-tunes a small pretrained
encoder on the same data splits as everything else. The checkpoint changed
twice in v1.5: `bert-mini` -> `distilbert-base-uncased` (matching another
branch's choice, at the user's request) -> `google/bert_uncased_L-2_H-128_A-2`
("BERT-tiny", 2 layers, hidden 128, ~4.4M params - smaller than bert-mini),
saved to `model_store/bert_tiny_v1/`.

**Rationale.** `distilbert-base-uncased` (~66M params) turned out
impractical on this CPU-only setup: a smoke test extrapolated to roughly
20+ hours for a single epoch on the full ~400k-row dataset, versus
`bert-mini`'s 1.5-3 hours. Swapping to BERT-tiny cut that to well under an
hour (confirmed: full run finished in ~41 minutes) while still being a
genuinely different, smaller-than-bert-mini transformer checkpoint. Using
the same splits and scoring keeps its macro-F1 directly comparable to the
other three models (§21's tuning table has the full distilbert timing
experiment).

**Known limitation.** BERT-tiny scored the lowest macro-F1 of the four
models (0.8519) despite having the *highest* ROC-AUC (0.9704, §22) - a real,
sensible split: ROC-AUC measures ranking quality regardless of threshold,
while macro-F1 measures classification quality at the threshold actually
used. A 2-layer, 1-epoch model this small is likely underfit at the
decision boundary even though its relative confidence ranking is good.

**Alternatives.** Frozen sentence embeddings are cheaper but usually score
below a full fine-tune. A hosted third-party API was ruled out: not open
source, costs per call, and can't be version-controlled.

## 16. MLflow for experiment tracking (v1.2)

**Context.** Before this, a training run just produced a file, with nothing
recording which settings produced which score.

**Decision.** Every run is now logged to MLflow - its settings, its scores,
files like the model and confusion matrix, and a `pip freeze` snapshot of
the exact library versions installed at the time - stored locally in
`mlflow.db`, no external server needed.

**Rationale.** DVC (§9) tracks which data went in; MLflow tracks which
settings, code, and library versions produced a given score. Together they
make any run reproducible from its logged record alone (seeds are fixed, so
re-running gives the same result). `requirements.txt` doesn't pin exact
versions, so the `pip freeze` snapshot is what makes a library-version
change visible after the fact, even though it can't prevent one.

**Hypothesis and conclusion tags (v1.4).** Each run now also logs what it
expected to happen before training starts, and what actually happened once
the metrics were in, so that reasoning lives with the run itself.

**Alternatives.** Hosted tools like Weights & Biases need an account; this
project is meant to run fully offline.

## 17. Class imbalance — cost-sensitive loss, not resampling (v1.2)

**Context.** NEGATIVE reviews are a small minority of the data - ~13.5%
under the current binary VADER labels (§3) - so a model left to its own
devices tends to mostly ignore that class.

**Decision.** All four models weight the rare class more heavily during
training: `logreg` and `linear_svc` via `class_weight="balanced"`, the RNN
via a weighted `CrossEntropyLoss`, and the transformer via a custom
`WeightedTrainer` that overrides `compute_loss` (HuggingFace's stock
`Trainer` doesn't support this out of the box). Briefly changed (v1.5):
`logreg` dropped this weighting, at the user's explicit request, matching
another branch's config - the measurement below is what that experiment
found, and why it was reverted (§21).

**Rationale.** The effect is large and directly measurable. With weighting,
NEGATIVE recall is 0.9140 (`logreg`), 0.7026 (`linear_svc`), 0.8812
(`bert_tiny`), and 0.9324 (`rnn_lstm`) - `rnn_lstm` catches 93% of NEGATIVE
reviews on the same data. Dropping `logreg`'s weighting (tried, then
reverted, v1.5) cut its NEGATIVE recall to 0.6698 - confirming the same
pattern the other three models already showed. The tradeoff is precision:
`logreg`'s NEGATIVE precision (0.604) is the lowest of the four weighted
models, since weighting trades some false positives for far fewer missed
NEGATIVE reviews.

**Alternatives.** Duplicating rare rows (oversampling) has the same effect
but is slower. Removing common rows would throw away real data. SMOTE
doesn't suit this kind of high-dimensional sparse text data.

## 18. Engineering-grade run records (v1.2)

**Context.** §16 got runs into MLflow, but the linear and transformer
trainers logged very different amounts of detail, making them hard to
compare fairly.

**Decision.** Every run now logs the same things - a hash of the exact data
used, the same metric names (including ROC-AUC), and whether the code had
uncommitted changes - and `training/compare_runs.py` prints a ranked
leaderboard in the terminal.

**Rationale.** A comparison is only as useful as its weakest entry; logging
the same things for every model is what makes them comparable at all.

**Alternatives.** Logging the entire dataset with every run was ruled out -
too much storage for no benefit, since a hash already proves it matches.

## 19. Three training stages, correctly named (v1.2)

**Context.** `linear_svc` only ran behind a manual `--model` flag, so it
never showed up from a plain `dvc repro`. Separately, the transformer's
saved folder was still named `distilbert_v1`, left over from an earlier
model it no longer trains.

**Decision.** Added a real `train_linear_svc` stage to `dvc.yaml`. Renamed
the transformer's output folder to `bert_mini_v1`.

**Rationale.** A model that only runs if someone remembers a flag isn't
really part of the pipeline. A wrong artifact name is worse than no name -
it actively misleads whoever opens it.

## 20. A recurrent model as a third baseline (v1.2)

**Context.** The comparison had two extremes: a simple bag-of-words model
(§13) and a full pretrained transformer (§15). A recurrent net (RNN/LSTM)
sits in between - it can learn word order, but starts with no pretraining.

**Decision.** `training/train_rnn.py` trains an RNN/LSTM from scratch,
building its own vocabulary, on the same data splits and same class
weighting as the linear baseline.

**Rationale.** It needed its own module because its input (an ordered
sequence of words) is a different shape than the linear model's bag-of-words
features - reusing that vectorizer would mean densifying a huge sparse
matrix for no reason.

**Alternatives.** Built in PyTorch, not TensorFlow/Keras: TensorFlow doesn't
publish wheels for the Python version this project runs on, and PyTorch was
already a dependency for the transformer stage.

## 21. Tuning experiments — kept vs. reverted (v1.2 / v1.3 / v1.4 / v1.5)

**Context.** Every row below is a real, measured experiment judged by
macro-F1 (§14), not a guess, so a dropped idea doesn't get tried again by
accident.

| Experiment | Change | Result | Decision | Version |
|---|---|---|---|---|
| TF-IDF vocabulary width | `max_features` 20,000 → 30,000 with trigrams `(1,3)` | `saga` failed to converge; macro-F1 fell to 0.537 | Reverted | v1.2 |
| TF-IDF vocabulary width (retry) | `max_features` 20,000 → 24,000, bigrams `(1,2)` | Still non-convergent; macro-F1 0.556 | Reverted to the original 20,000 / `(1,2)` | v1.2 |
| RNN epochs | 3 → 4 | val_loss stopped improving after epoch 3; macro-F1 fell 0.6436 → 0.6393 | Reverted to 3 | v1.2 |
| RNN epochs (re-verified on the post-bugfix, larger dataset) | 3 vs. 4 vs. 6 | 3 → 0.6488, 4 → 0.6334, 6 → 0.6438 — 3 still wins, and the dip-then-partial-recovery pattern reproduced identically (same loss values) across separate runs, confirming training is deterministic | Kept at 3 | v1.3 |
| Sentiment thresholds | Scheme A (`<6`/`6–8`/`≥8`) → NPS-style (`<7`/`7–9`/`≥9`) | Macro-F1 improved on every model | Reverted — prioritizes accuracy over macro-F1 as the headline metric, a deliberate call, not a measurement failure | v1.2 |
| Transformer input text | `clean_review` → raw `full_review` | Macro-F1 0.6461 → 0.6459 — a wash, likely masked by `MAX_LENGTH=64` truncating before the difference could show | Reverted to `clean_review` | v1.2 |
| `linear_svc` calibration | Wrapped in `CalibratedClassifierCV` (sigmoid, cv=5) to get real `predict_proba` output | Accuracy improved 0.7206 → 0.7356; macro-F1 fell 0.6225 → 0.6049 | Kept the calibrated version in the codebase as a real, measured serving candidate — but not selected for serving (§22), since it trades away macro-F1 | v1.4 |
| Transformer checkpoint size | `bert-mini` → `distilbert-base-uncased` (~66M params) | Smoke test extrapolated to ~20+ hours for one epoch on the full dataset, on CPU with no GPU available — far past what's workable here | Reverted to a smaller checkpoint — see the next row, not back to bert-mini | v1.5 |
| Transformer checkpoint size (retry) | `distilbert-base-uncased` → `google/bert_uncased_L-2_H-128_A-2` ("BERT-tiny", ~4.4M params, smaller than bert-mini) | Full run finished in ~41 minutes; macro-F1 0.8519, the lowest of the four current models, but ROC-AUC 0.9704, the *highest* — see §15's known limitation | Kept — the only checkpoint size that was actually practical to train fully here | v1.5 |
| `logreg` class weighting | Dropped `class_weight="balanced"` (solver `saga`→`lbfgs`), at the user's explicit request, matching another branch's config | NEGATIVE recall fell to 0.6698, well below what the other three weighted models get (0.70-0.93, §17) | Reverted — `class_weight="balanced"` restored once the recall drop was measured | v1.5 |

**How to read a "reverted" row.** It means the code went back to what it was
before - not that the experiment was wasted. A negative result is still
worth keeping on record, same idea as §12's cleaning diagnostics.

## 22. Serving `rnn_lstm`, the highest-scoring model on current data (v1.4, re-confirmed v1.5)

**Context.** The API serves one model out of four, picked by macro-F1
(§14). Both the labeling scheme (§3, now binary VADER) and every model's
config changed since this decision was first made (v1.4) - re-measured
after all four retrained on the current data (v1.5) to check whether the
served-model choice still holds.

**Decision.** `serving/app.py` (FastAPI) serves `rnn_lstm` - still the
answer after retraining. It validates input (rejects empty, whitespace-only,
or junk-only text), loads the model once at startup, and returns a
prediction with confidence, per-class probabilities, latency, and which
model version answered (`model_version`, added v1.5).

**The final four-way comparison, on the current binary VADER-labeled data:**

| Model | macro-F1 | accuracy | ROC-AUC |
|---|---|---|---|
| **`rnn_lstm`** | **0.8918** | 0.9434 | — |
| `linear_svc` (calibrated) | 0.8652 | 0.9412 | 0.9671 |
| `bert_tiny` | 0.8519 | 0.9208 | 0.9704 |
| `logreg` | 0.8358 | 0.9073 | 0.9690 |

`rnn_lstm` wins on macro-F1 - the metric this project has used throughout
(§14) - by a clear margin over the next-best model (0.8918 vs 0.8652), not
a close call. This is a different picture from the older, closer 3-class
result (§13/§17), where the gap between models was smaller: on the binary,
text-derived labels, `rnn_lstm`'s ability to read word order pulls further
ahead, especially on NEGATIVE-class recall (0.9324, vs 0.70-0.91 for the
other three, §17).

**Why not the others.** `linear_svc` scores higher on plain accuracy
(94.12%) than `rnn_lstm` at first glance, but the two are close enough
(94.12% vs 94.34%) that this isn't actually the deciding factor here -
`rnn_lstm` wins on both accuracy *and* macro-F1 this time, unlike the
closer tradeoffs measured elsewhere in this project (§21). `bert_tiny` has
the *highest* ROC-AUC (0.9704) but not the highest macro-F1 - a real,
explained split (§15's known limitation), not a contradiction; it just
isn't the best classifier at the threshold actually used, even though its
relative confidence ranking is good. `logreg` scores the lowest macro-F1 of
the four despite being weighted the same way as the others (§17) - its
NEGATIVE recall (0.9140) is close to `rnn_lstm`'s, but that comes at the
cost of the lowest NEGATIVE precision of the four (0.604), pulling its
macro-F1 down.

**Docker packaging.** The root `Dockerfile` builds the API into a single
container: a `python:3.12-slim` base, a scoped `serving/requirements.txt`
instead of the full project one (no `dvc`, `mlflow`, or `scikit-learn` -
`rnn_lstm` doesn't need them to serve), and only the files the API actually
uses copied in (`serving/`, `features/build_features.py` for `clean_text()`,
and the trained RNN's weights and vocabulary file). `rnn_lstm` doesn't need
`transformers` either - just `torch` and a small JSON vocabulary. Anyone
with Docker can run the API the same way, without setting up a matching
Python environment by hand.

**Response contract hardening (v1.5).** Two additions, prompted by comparing
against a reference implementation covering similar serving material with a
different (churn prediction) example: `model_version` is now returned on
every response, so a caller can tell two deployments apart without reading
server logs; and
`confidence` is now constrained with `Field(..., ge=0.0, le=1.0)` on the
*response* itself, not just request inputs - if a future model swap ever
returned something outside that range, Pydantic would reject the response
with a loud `500` instead of silently handing a caller a nonsensical value.

**Calibration experiment on `linear_svc`.** Before this, `linear_svc` was
tested as a middle ground. Giving it real confidence scores (via
`CalibratedClassifierCV`) raised its accuracy further but lowered its
macro-F1 relative to an uncalibrated baseline (§21) - the same
accuracy-vs-macro-F1 tradeoff this project keeps measuring, though on the
current binary data it's no longer the deciding factor against `rnn_lstm`.

**Alternatives.** `bert_tiny` was tried as the served model's counterpart
question (is the transformer worth the dependency weight) and lost on
macro-F1 despite winning on ROC-AUC, as explained above. A version of the
API that lets callers pick any of the four models was considered and
rejected as unnecessary complexity for a first version.

## 23. A plain HTML page over the API, not a framework app (v1.4)

**Context.** The API (§22) only speaks JSON over HTTP - useful for another
program, but not something a person can just click around in. A simple way
to type a review and see the predicted sentiment was worth having on top of
it.

**Decision.** `ui/index.html` is one self-contained file: a text box, an
"Analyze" button, and a result showing the sentiment, the confidence
percentage, and a bar for each class's probability. Plain HTML, CSS, and
JavaScript - no framework (like React) and no build step.

**Rationale.** The whole page is one form talking to one API endpoint, which
doesn't need a framework's complexity to build. Anyone can open the file (or
serve the folder) and it just works, with nothing to install or compile
first.

**A consequence: CORS.** The UI page and the API run on two different local
addresses (different ports). Browsers block a page from calling a different
address like that by default, as a security measure - otherwise a malicious
page could quietly call APIs it has no business touching. Since the UI
calling the API here is intentional, `serving/app.py` explicitly allows it
(`CORSMiddleware`) - without that, the page would load fine, but every
"Analyze" click would fail silently in the browser.
