"""Pure training/evaluation logic — no MLflow, no experiment config.

Importable by experiment orchestration (`run_experiment.py`) and by serving.
Every function is a plain, testable transformation.
"""

import sqlite3

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from feature_store.feature_store import REPO_ROOT

FEATURE_STORE_PATH = REPO_ROOT / "feature_store" / "feature_store.db"
LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

DEFAULT_TFIDF_PARAMS = dict(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
)


def load_data():
    """Load (clean_review, sentiment) from the feature store.

    Returns: X (text Series), y (encoded labels), label_encoder.
    """
    conn = sqlite3.connect(FEATURE_STORE_PATH)
    df = pd.read_sql("SELECT clean_review, sentiment FROM hotel_review_features", conn)
    conn.close()
    df = df.dropna(subset=["clean_review", "sentiment"]).reset_index(drop=True)
    le = LabelEncoder()
    y = le.fit_transform(df["sentiment"])
    return df["clean_review"], y, le


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split (classes are imbalanced)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_model(estimator=None, tfidf_params=None) -> Pipeline:
    """TF-IDF + classifier pipeline. Pass any sklearn estimator to swap models."""
    if estimator is None:
        estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    params = {**DEFAULT_TFIDF_PARAMS, **(tfidf_params or {})}
    return Pipeline([("tfidf", TfidfVectorizer(**params)), ("clf", estimator)])


def evaluate(model, X_test, y_test, label_names=LABEL_NAMES):
    """Score a fitted model.

    Returns: (metrics dict, confusion-matrix DataFrame, classification-report text).
    ROC-AUC is only included if the estimator exposes predict_proba.
    """
    y_pred = model.predict(X_test)
    report_dict = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True
    )
    report_text = classification_report(y_test, y_pred, target_names=label_names)

    metrics = {"accuracy": accuracy_score(y_test, y_pred),
               "macro_f1": report_dict["macro avg"]["f1-score"]}
    for label in label_names:
        r = report_dict[label]
        metrics[f"{label}_precision"] = r["precision"]
        metrics[f"{label}_recall"] = r["recall"]
        metrics[f"{label}_f1"] = r["f1-score"]

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        metrics["roc_auc_macro"] = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="macro"
        )

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in label_names],
        columns=[f"pred_{l}" for l in label_names],
    )
    return metrics, cm_df, report_text
