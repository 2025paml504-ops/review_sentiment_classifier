from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import load_config, resolve_path
from .data import prepare_data
from .text import simple_tokens


def make_estimator(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed)
    if name == "linear_svm":
        return CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=seed), cv=3)
    if name == "naive_bayes":
        return MultinomialNB(alpha=0.5)
    raise ValueError(f"Unknown model: {name}")


def make_pipeline(config: dict, model_name: str) -> Pipeline:
    features = config["features"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    max_features=int(features["max_features"]),
                    ngram_range=tuple(features["ngram_range"]),
                    min_df=features["min_df"],
                    max_df=features["max_df"],
                    sublinear_tf=True,
                ),
            ),
            ("classifier", make_estimator(model_name, int(config["seed"]))),
        ]
    )


def evaluate(y_true, y_pred) -> tuple[dict[str, float], dict]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return metrics, report


def build_reference(model: Pipeline, texts: pd.Series, predictions: np.ndarray) -> dict:
    vocab = sorted(model.named_steps["tfidf"].vocabulary_)
    lengths = np.array([len(simple_tokens(text)) for text in texts], dtype=float)
    upper = max(10.0, float(np.percentile(lengths, 99)) + 1)
    bins = np.unique(np.linspace(0, upper, 11)).tolist()
    if len(bins) < 3:
        bins = [0.0, 5.0, 10.0]
    counts, _ = np.histogram(lengths, bins=bins)
    distribution = pd.Series(predictions).value_counts(normalize=True).sort_index().to_dict()
    return {
        "vocabulary": vocab,
        "length_bins": bins,
        "length_distribution": (counts / max(counts.sum(), 1)).tolist(),
        "prediction_distribution": distribution,
        "training_rows": int(len(texts)),
    }


def train(config: dict) -> dict:
    processed = resolve_path("data/processed")
    if not (processed / "train.csv").exists() or not (processed / "test.csv").exists():
        prepare_data(config)
    data_cfg = config["data"]
    train_df = pd.read_csv(processed / "train.csv")
    test_df = pd.read_csv(processed / "test.csv")
    x_train, y_train = train_df[data_cfg["text_column"]], train_df[data_cfg["label_column"]]
    x_test, y_test = test_df[data_cfg["text_column"]], test_df[data_cfg["label_column"]]

    tracking_uri = str(resolve_path(config["training"]["mlflow_tracking_uri"]))
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(Path(tracking_uri).as_uri())
    mlflow.set_experiment(config["training"]["experiment_name"])
    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for model_name in config["training"]["models"]:
        with mlflow.start_run(run_name=model_name) as run:
            model = make_pipeline(config, model_name)
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            metrics, report = evaluate(y_test, predicted)
            mlflow.log_params(
                {
                    "model": model_name,
                    "seed": config["seed"],
                    "max_features": config["features"]["max_features"],
                    "ngram_range": str(config["features"]["ngram_range"]),
                    "dataset_sha256": json.loads(
                        (processed / "validation_report.json").read_text(encoding="utf-8")
                    )["sha256"],
                }
            )
            mlflow.log_metrics(metrics)
            results[model_name] = {**metrics, "run_id": run.info.run_id, "report": report}
            fitted[model_name] = model

    metric = config["training"]["primary_metric"]
    champion_name = max(results, key=lambda name: results[name][metric])
    champion = fitted[champion_name]
    model_dir = resolve_path("models")
    report_dir = resolve_path("reports")
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, model_dir / "model.joblib")

    champion_predictions = champion.predict(x_test)
    labels = sorted(y_train.unique())
    metadata = {
        "model_name": champion_name,
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "metrics": {k: v for k, v in results[champion_name].items() if isinstance(v, float)},
        "dataset": json.loads((processed / "validation_report.json").read_text(encoding="utf-8")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    serializable_results = {
        name: {k: v for k, v in result.items() if k != "report"} for name, result in results.items()
    }
    comparison = {
        "champion": champion_name,
        "primary_metric": metric,
        "models": serializable_results,
    }
    (report_dir / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    matrix = confusion_matrix(y_test, champion_predictions, labels=labels).tolist()
    (report_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "labels": labels,
                "confusion_matrix": matrix,
                "report": results[champion_name]["report"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    reference = build_reference(champion, x_train, champion.predict(x_train))
    (processed / "reference.json").write_text(json.dumps(reference), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare text classifiers")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    result = train(load_config(args.config))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
