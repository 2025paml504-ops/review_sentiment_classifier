import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

from features.build_features import SENTIMENT_LABELS
from features.vectorize import RANDOM_STATE, TEST_CSV, TRAIN_CSV, VECTORIZER_PATH
from training import tracking

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_linear")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_STORE = REPO_ROOT / "model_store"
METRICS_DIR = Path(__file__).resolve().parent
METRICS_PATH = METRICS_DIR / "metrics_logreg.json"

MODELS = {
    "logreg": (
        lambda: LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        ),
        "logreg_v1.pkl",
    ),
    "linear_svc": (
        lambda: LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
        "linear_svc_v1.pkl",
    ),
}

# Recurrent models (RNN / LSTM / BiLSTM) deliberately do **not** live here:
# they consume an ordered sequence of token ids, not the TF-IDF matrix this
# module is built around. See training/train_rnn.py (v1.2).

DEFAULT_MODEL = "logreg"


def metrics_path(model_name: str) -> Path:
    """The git-tracked metrics file DVC watches for this model."""
    if model_name == DEFAULT_MODEL:
        return METRICS_PATH
    return METRICS_DIR / f"metrics_{model_name}.json"


def load_split(path: Path, limit: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `dvc repro vectorize` first")
    df = pd.read_csv(path, nrows=limit)
    df["clean_review"] = df["clean_review"].fillna("").astype(str)
    return df


def build_model(name: str = DEFAULT_MODEL):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}', expected one of {sorted(MODELS)}")
    return MODELS[name][0]()


def decision_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    return model.decision_function(X_test)


def roc_auc(model, X_test, y_test) -> dict:
    scores = decision_scores(model, X_test)
    labels = list(model.classes_)
    aucs = {}
    for average in ("macro", "weighted"):
        try:
            aucs[f"roc_auc_{average}"] = float(
                roc_auc_score(y_test, scores, multi_class="ovr", average=average, labels=labels)
            )
        except ValueError as exc:
            logger.warning("ROC-AUC (%s) not computable: %s", average, exc)
    return aucs


# Added (v1.2): a forced 3-way call caps accuracy on this data - the
# NEUTRAL band is genuinely ambiguous *text*, not just a fuzzy label, so no
# amount of tuning gets a forced guess much past ~67% (see decisions.md #17/#3).
# Letting the model abstain below a confidence threshold instead of guessing
# raises accuracy on what it does answer, at the cost of leaving the least
# confident reviews unclassified. Confirmed empirically: threshold 0.6 on the
# unweighted logreg covers 62% of reviews at 76.5% accuracy on that subset.
def evaluate(model, X_test, y_test, confidence_threshold: float | None = None) -> dict:
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, labels=SENTIMENT_LABELS, output_dict=True, zero_division=0
    )
    metrics = {
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(report["accuracy"]),
        **roc_auc(model, X_test, y_test),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in SENTIMENT_LABELS
        },
        "confusion_matrix": {
            "labels": SENTIMENT_LABELS,
            "rows_true_cols_pred": confusion_matrix(
                y_test, y_pred, labels=SENTIMENT_LABELS
            ).tolist(),
        },
    }

    if confidence_threshold is not None:
        if not hasattr(model, "predict_proba"):
            logger.warning(
                "confidence_threshold requested but %s has no predict_proba; skipping",
                type(model).__name__,
            )
        else:
            proba = model.predict_proba(X_test)
            confidence = proba.max(axis=1)
            thresholded_pred = model.classes_[proba.argmax(axis=1)]
            covered = confidence >= confidence_threshold
            y_test_arr = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test
            metrics["confidence_threshold"] = confidence_threshold
            metrics["coverage"] = float(covered.mean())
            metrics["accuracy_at_threshold"] = (
                float(accuracy_score(y_test_arr[covered], thresholded_pred[covered]))
                if covered.any()
                else None
            )
    return metrics


def run_params(
    model, model_name: str, limit: int | None, confidence_threshold: float | None = None
) -> dict:
    params = {}
    if model is not None and hasattr(model, "get_params"):
        params.update({f"model.{k}": v for k, v in model.get_params().items()})
    params.update(
        {
            "model_name": model_name,
            "random_state": RANDOM_STATE,
            "limit": limit if limit is not None else "none",
            "train_csv": TRAIN_CSV.name,
            "test_csv": TEST_CSV.name,
            "vectorizer": VECTORIZER_PATH.name,
            # Added (v1.2)
            "confidence_threshold": confidence_threshold if confidence_threshold is not None else "none",
        }
    )
    return params


def train(
    model_name: str = DEFAULT_MODEL,
    limit: int | None = None,
    confidence_threshold: float | None = None,  # Added (v1.2)
) -> dict:
    train_df = load_split(TRAIN_CSV, limit)
    test_df = load_split(TEST_CSV, limit)
    vectorizer = joblib.load(VECTORIZER_PATH)

    X_train = vectorizer.transform(train_df["clean_review"])
    X_test = vectorizer.transform(test_df["clean_review"])
    y_train = train_df["sentiment"]
    y_test = test_df["sentiment"]

    logger.info("train %s, test %s, model %s", X_train.shape, X_test.shape, model_name)

    model = build_model(model_name)
    tags = {"stage": "train", "framework": "scikit-learn", "smoke_test": str(limit is not None)}

    params = run_params(model, model_name, limit, confidence_threshold)
    with tracking.start_run(model_name, params, tags) as run:
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, confidence_threshold)
        metrics["model"] = model_name
        metrics["n_train"] = int(X_train.shape[0])
        metrics["n_test"] = int(X_test.shape[0])
        metrics["n_features"] = int(X_train.shape[1])

        MODEL_STORE.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_STORE / MODELS[model_name][1]
        joblib.dump(model, model_path)

        run.log_metrics(metrics)
        run.log_dict(metrics["confusion_matrix"], "confusion_matrix.json")
        run.log_sklearn_model(model)
        run.log_artifact(VECTORIZER_PATH, "vectorizer")

        # Smoke runs (--limit) must not overwrite the scored metrics file DVC
        # tracks; the run itself is still recorded, tagged smoke_test=True.
        if limit is None:
            metrics_path(model_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            run.log_artifact(metrics_path(model_name))

        metrics["mlflow_run_id"] = run.run_id
        logger.info("MLflow run id: %s (experiment %s)", run.run_id, tracking.EXPERIMENT_NAME)

    logger.info("macro-F1 %.4f, accuracy %.4f", metrics["macro_f1"], metrics["accuracy"])
    # Added (v1.2)
    if "accuracy_at_threshold" in metrics and metrics["accuracy_at_threshold"] is not None:
        logger.info(
            "confidence >= %.2f: coverage %.1f%%, accuracy %.4f",
            metrics["confidence_threshold"],
            100 * metrics["coverage"],
            metrics["accuracy_at_threshold"],
        )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    parser.add_argument("--limit", type=int, default=None, help="only read the first N rows")
    # Added (v1.2): optional abstention - only count a prediction when
    # confident, and report coverage + accuracy on the covered subset instead of
    # forcing a guess on every review.
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="abstain below this predicted-probability confidence; logs coverage and accuracy_at_threshold",
    )
    args = parser.parse_args()
    train(args.model, args.limit, args.confidence_threshold)
