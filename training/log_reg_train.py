"""Train and evaluate the TF-IDF + LogisticRegression sentiment model.

Loads the pre-fit TF-IDF vectorizer and train/test splits produced by
features/vectorize.py — does not refit TF-IDF, so train/serve vectorization
stays identical.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from feature_store.feature_store import REPO_ROOT
from training.run_experiment import evaluate

TRAIN_CSV = REPO_ROOT / "data" / "processed" / "train_v1.csv"
TEST_CSV = REPO_ROOT / "data" / "processed" / "test_v1.csv"
VECTORIZER_PATH = REPO_ROOT / "model_store" / "tfidf_vectorizer_v1.pkl"

MODEL_PATH = REPO_ROOT / "model_store" / "sentiment_logreg_v1.pkl"
METRICS_PATH = REPO_ROOT / "metrics" / "train_metrics.json"
CONFUSION_MATRIX_PATH = REPO_ROOT / "metrics" / "confusion_matrix.csv"


def main() -> None:
    print(f"Loading fitted vectorizer: {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)

    print(f"Loading train split: {TRAIN_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    X_train_vec = vectorizer.transform(train_df["clean_review"].fillna(""))
    y_train = train_df["sentiment"]

    print(f"Loading test split: {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV)
    X_test_vec = vectorizer.transform(test_df["clean_review"].fillna(""))
    y_test = test_df["sentiment"]

    print(f"Train: {X_train_vec.shape}  Test: {X_test_vec.shape}")

    print("Training LogisticRegression on pre-vectorized features...")
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluating...")
    metrics, cm_df, report_text = evaluate(model, X_test_vec, y_test)

    print("\nConfusion Matrix:")
    print(cm_df)
    print("\nClassification Report:")
    print(report_text)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    # Only the classifier is saved here — the fitted vectorizer already
    # lives at VECTORIZER_PATH (produced by vectorize.py). Anyone loading
    # this model for inference must also load that vectorizer.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {METRICS_PATH}")

    cm_df.to_csv(CONFUSION_MATRIX_PATH)
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()