"""TF-IDF vectorization of clean_review, fit on the train split only (no leakage)."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from feature_store.feature_store import load_features

# Repo root is two levels up from this file (features/vectorize.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = REPO_ROOT / "data" / "processed" / "train_v1.csv"
VECTORIZER_PATH = REPO_ROOT / "model_store" / "tfidf_vectorizer_v1.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# TF-IDF hyperparameters.
MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
MIN_DF = 5
SUBLINEAR_TF = True


def build_vectorizer() -> TfidfVectorizer:
    """Construct a TfidfVectorizer with the project defaults."""
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=SUBLINEAR_TF,
    )


def build_tfidf_features(test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split, fit TF-IDF on train only, and return the vectorized splits + vectorizer.

    Returns: X_train_vec, X_test_vec, y_train, y_test, vectorizer
    """
    df = load_features()
    X = df["clean_review"].fillna("")
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Persist the train split for reproducibility/versioning.
    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train_df = X_train.to_frame(name="clean_review")
    train_df["sentiment"] = y_train
    train_df.to_csv(TRAIN_CSV, index=False)

    # Fit on train only to avoid leaking test vocabulary/idf into features.
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


if __name__ == "__main__":
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = build_tfidf_features()
    print("X_train:", X_train_vec.shape)
    print("X_test:", X_test_vec.shape)
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("Wrote train split:", TRAIN_CSV)
    print("Wrote vectorizer:", VECTORIZER_PATH)
