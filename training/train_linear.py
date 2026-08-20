"""Train a TF-IDF + linear model (logreg or linear_svc) on hotel reviews.

How this file is organized, top to bottom, following the same order the
code actually runs in:

    - Config           - which models exist, and what result each one is
      expected to produce (MODELS, HYPOTHESES)
    - Data loading      - load_split()
    - Model building    - build_model()
    - Scoring helpers   - decision_scores(), roc_auc(), evaluate()
    - train()           - the actual pipeline: load -> vectorize -> fit ->
      evaluate -> save. This is the function that ties everything above
      together, and the one thing you need to read if you only read one
      function here.
    - CLI               - what happens when you run this file directly

Run it:

    python -m training.train_linear                     # trains logreg (the default)
    python -m training.train_linear --model linear_svc   # trains linear_svc instead
    python -m training.train_linear --limit 5000         # quick smoke test on 5000 rows
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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


# ── Config ────────────────────────────────────────────────────────────
# Recurrent models (RNN / LSTM / BiLSTM) deliberately do **not** live here:
# they consume an ordered sequence of token ids, not the TF-IDF matrix this
# module is built around. See training/train_rnn.py.

DEFAULT_MODEL = "logreg"

# Each entry: model_name -> (a function that builds a fresh, untrained
# model, the filename to save it under in model_store/).
MODELS = {
    # class_weight="balanced" (restored v1.5) - v1.4 briefly dropped this to
    # match task-2-branch-sonal-tmp's unweighted config, but unweighted
    # training under-catches the small NEGATIVE class on this imbalanced
    # data; see decisions.md #17 for the measured comparison.
    "logreg": (
        lambda: LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "logreg_v1.pkl",
    ),
    # Wrapped in CalibratedClassifierCV: plain LinearSVC has no
    # predict_proba, only decision_function (a raw margin, not a
    # probability). Calibration fits 5 LinearSVC clones under
    # cross-validation and turns their scores into real per-class
    # probabilities - done on the training data only, so it can't leak into
    # the reported test metrics. See decisions.md #21/#22 for what this
    # traded off (higher accuracy, lower macro-F1).
    "linear_svc": (
        lambda: CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
            method="sigmoid",
            cv=5,
        ),
        "linear_svc_v1.pkl",
    ),
}

# What each config is expected to show, logged as an MLflow tag before
# training starts and checked against the real result afterward
# (training/tracking.py's start_run/set_conclusion).
HYPOTHESES = {
    "logreg": (
        "Restoring class_weight='balanced' will raise NEGATIVE recall back "
        "above the unweighted config, at some cost to raw accuracy - "
        "matching decisions.md #17's original finding."
    ),
    "linear_svc": (
        "Calibrating LinearSVC's decision_function into probabilities "
        "will raise accuracy but lower macro-F1 - the same "
        "accuracy-vs-macro-F1 tradeoff this project keeps measuring "
        "elsewhere (decisions.md #14, #22)."
    ),
}


def metrics_path(model_name: str) -> Path:
    """The git-tracked metrics file DVC watches for this model."""
    if model_name == DEFAULT_MODEL:
        return METRICS_PATH
    return METRICS_DIR / f"metrics_{model_name}.json"


# ── Data loading ──────────────────────────────────────────────────────

def load_split(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Read a train/test split written by features/vectorize.py."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `dvc repro vectorize` first")
    df = pd.read_csv(path, nrows=limit)
    df["clean_review"] = df["clean_review"].fillna("").astype(str)
    return df


# ── Model building ────────────────────────────────────────────────────

def build_model(name: str = DEFAULT_MODEL):
    """Look up `name` in MODELS and construct a fresh, untrained estimator."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}', expected one of {sorted(MODELS)}")
    return MODELS[name][0]()


# ── Scoring helpers ───────────────────────────────────────────────────

def decision_scores(model, X_test):
    """Per-class confidence scores: real probabilities if the model has
    them, otherwise the raw decision-boundary margin as a fallback."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    return model.decision_function(X_test)


def roc_auc(model, X_test, y_test) -> dict:
    """ROC-AUC, computed differently for 2 classes vs. 3+.

    Binary only needs the positive class's score, and there's nothing to
    average - macro and weighted come out identical. 3+ classes need
    scikit-learn's one-vs-rest averaging instead, which expects the full
    per-class score matrix. Mixing the two up is what used to raise
    "y should be a 1d array, got shape (N, 2)" here before this branch
    existed (v1.4).
    """
    scores = decision_scores(model, X_test)
    labels = list(model.classes_)

    if len(labels) == 2:
        positive_scores = scores[:, 1] if scores.ndim == 2 else scores
        try:
            auc = float(roc_auc_score(y_test, positive_scores))
        except ValueError as exc:
            logger.warning("ROC-AUC not computable: %s", exc)
            return {}
        return {"roc_auc_macro": auc, "roc_auc_weighted": auc}

    aucs = {}
    for average in ("macro", "weighted"):
        try:
            aucs[f"roc_auc_{average}"] = float(
                roc_auc_score(y_test, scores, multi_class="ovr", average=average, labels=labels)
            )
        except ValueError as exc:
            logger.warning("ROC-AUC (%s) not computable: %s", average, exc)
    return aucs


def evaluate(model, X_test, y_test, confidence_threshold: float | None = None) -> dict:
    """Score a fitted model: macro/weighted F1, accuracy, ROC-AUC, a
    per-class breakdown, and a confusion matrix.

    `confidence_threshold` is optional abstention: instead of forcing a
    guess on every review, only count predictions the model is confident
    enough about, and separately report how much of the data that covers
    and how accurate it is on that covered subset. Useful because some
    reviews are genuinely ambiguous or borderline even under the current
    text-derived labels (decisions.md #3 has a real example: a review
    reading "staff rude unhelpful" that a numeric score alone called
    positive) - not every low-confidence prediction is the model being wrong.
    """
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
    """The settings worth logging to MLflow for this run."""
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
            "confidence_threshold": confidence_threshold if confidence_threshold is not None else "none",
        }
    )
    return params


# ── train() - the actual pipeline ────────────────────────────────────

def train(
    model_name: str = DEFAULT_MODEL,
    limit: int | None = None,
    confidence_threshold: float | None = None,
) -> dict:
    """Load data -> vectorize -> fit -> evaluate -> save. One MLflow run per call."""

    # Step 1: load the pre-split, pre-vectorized data.
    # The TF-IDF vectorizer was already fit once in features/vectorize.py -
    # here we only *use* it (`.transform`), never re-fit it, so train and
    # test features come from the exact same vocabulary.
    train_df = load_split(TRAIN_CSV, limit)
    test_df = load_split(TEST_CSV, limit)
    vectorizer = joblib.load(VECTORIZER_PATH)

    X_train = vectorizer.transform(train_df["clean_review"])
    X_test = vectorizer.transform(test_df["clean_review"])
    y_train = train_df["sentiment"]
    y_test = test_df["sentiment"]
    logger.info("train %s, test %s, model %s", X_train.shape, X_test.shape, model_name)

    # Step 2: build the (untrained) model.
    model = build_model(model_name)

    # Step 3: fit, evaluate, and log everything to MLflow in one tracked run.
    tags = {"stage": "train", "framework": "scikit-learn", "smoke_test": str(limit is not None)}
    params = run_params(model, model_name, limit, confidence_threshold)

    with tracking.start_run(model_name, params, tags, hypothesis=HYPOTHESES.get(model_name)) as run:
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, confidence_threshold)
        metrics["model"] = model_name
        metrics["n_train"] = int(X_train.shape[0])
        metrics["n_test"] = int(X_test.shape[0])
        metrics["n_features"] = int(X_train.shape[1])

        # Step 4: save the model file itself.
        MODEL_STORE.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_STORE / MODELS[model_name][1]
        joblib.dump(model, model_path)

        # Step 5: log the run's record - metrics, files, and what actually
        # happened compared to the hypothesis logged at the start.
        run.log_metrics(metrics)
        run.log_dict(metrics["confusion_matrix"], "confusion_matrix.json")
        run.log_sklearn_model(model)
        run.log_artifact(VECTORIZER_PATH, "vectorizer")
        run.set_conclusion(
            f"macro-F1 {metrics['macro_f1']:.4f}, accuracy {metrics['accuracy']:.4f}, "
            f"NEGATIVE recall {metrics['per_class']['NEGATIVE']['recall']:.4f}."
        )

        # A smoke run (--limit) must not overwrite the scored metrics file
        # DVC tracks - the run itself is still recorded, just tagged
        # smoke_test=True so it doesn't get confused with a real result.
        if limit is None:
            metrics_path(model_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            run.log_artifact(metrics_path(model_name))

        metrics["mlflow_run_id"] = run.run_id
        logger.info("MLflow run id: %s (experiment %s)", run.run_id, tracking.EXPERIMENT_NAME)

    logger.info("macro-F1 %.4f, accuracy %.4f", metrics["macro_f1"], metrics["accuracy"])
    if "accuracy_at_threshold" in metrics and metrics["accuracy_at_threshold"] is not None:
        logger.info(
            "confidence >= %.2f: coverage %.1f%%, accuracy %.4f",
            metrics["confidence_threshold"],
            100 * metrics["coverage"],
            metrics["accuracy_at_threshold"],
        )
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    parser.add_argument("--limit", type=int, default=None, help="only read the first N rows")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="abstain below this predicted-probability confidence; logs coverage and accuracy_at_threshold",
    )
    args = parser.parse_args()
    train(args.model, args.limit, args.confidence_threshold)
