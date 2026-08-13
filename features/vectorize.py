"""TF-IDF vectorization of clean_review, fit on the train split only (no leakage).

Persists both train and test splits (raw text + labels) alongside the fitted
vectorizer, so downstream consumers (training/train.py) never need to
reconstruct the split independently.
"""

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from feature_store.feature_store import load_features
from features.tfidf_config import TFIDF_PARAMS

# Repo root is two levels up from this file (features/vectorize.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = REPO_ROOT / "data" / "processed" / "train_v1.csv"
TEST_CSV = REPO_ROOT / "data" / "processed" / "test_v1.csv"
VECTORIZER_PATH = REPO_ROOT / "model_store" / "tfidf_vectorizer_v1.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def build_vectorizer() -> TfidfVectorizer:
    """Construct a TfidfVectorizer using the shared project defaults."""
    return TfidfVectorizer(**TFIDF_PARAMS)


def build_tfidf_features(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series, TfidfVectorizer]:
    """Split, fit TF-IDF on train only, and return the vectorized splits + vectorizer.

    Returns:
        X_train_vec, X_test_vec, y_train, y_test, vectorizer
    """
    df = load_features()
    X = df["clean_review"].fillna("")
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Persist both splits (raw text + label) for reproducibility/versioning,
    # and so training/train.py can load the exact same test set rather than
    # re-deriving it.
    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train_df = X_train.to_frame(name="clean_review")
    train_df["sentiment"] = y_train
    train_df.to_csv(TRAIN_CSV, index=False)

    test_df = X_test.to_frame(name="clean_review")
    test_df["sentiment"] = y_test
    test_df.to_csv(TEST_CSV, index=False)

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
    print("Wrote test split:", TEST_CSV)
    print("Wrote vectorizer:", VECTORIZER_PATH)