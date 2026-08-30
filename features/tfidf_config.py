"""Shared TF-IDF hyperparameters, used by both the DVC vectorize stage
(features/vectorize.py) and the training pipeline (core.py's build_model).

Keeping this in one place ensures the vectorizer persisted to
model_store/tfidf_vectorizer_v1.pkl matches what's actually used at
training/serving time.
"""

TFIDF_PARAMS = dict(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=20000,
    sublinear_tf=True,
    min_df=5,
    max_df=0.95,
)