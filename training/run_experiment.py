"""Experiment orchestration: MLflow tracking wrapped around training.train.

Add a model/config to EXPERIMENTS and run this module — each config becomes one
MLflow run with its params, metrics, eval artifacts, and the fitted model logged.

    python -m training.run_experiment
"""

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

# from sklearn.naive_bayes import MultinomialNB   # example alternative model
# from sklearn.svm import LinearSVC                # note: no predict_proba -> no ROC-AUC

from feature_store.feature_store import REPO_ROOT
from training.experiment_train import (
    DEFAULT_TFIDF_PARAMS,
    build_model,
    evaluate,
    load_data,
    split_data,
)

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
mlflow.set_experiment("sentiment_classifier")


#--- Config runs ------

# ─── Define experiments here ─────────────────────────────────
# Each entry: run_name + an sklearn estimator (+ optional tfidf_params override).
# Add a new dict to try a different model or hyperparameters.


EXPERIMENTS = [
    {
        "run_name": "logreg_baseline",
        "estimator": LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    },
    {
        "run_name": "logreg_balanced",
        "estimator": LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced"
        ),
    },
    {
        # Label order from LabelEncoder (alphabetical): 0=NEGATIVE, 1=NEUTRAL, 2=POSITIVE.
        "run_name": "custom_weighting",
        "estimator": LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            class_weight={0: 2.0, 1: 1.5, 2: 1.0},
        ),
    },
    # {"run_name": "nb", "estimator": MultinomialNB()},
    # {"run_name": "svm_linear", "estimator": LinearSVC()},
    # {"run_name": "logreg_trigram", "estimator": LogisticRegression(max_iter=1000),
    #  "tfidf_params": {"ngram_range": (1, 3), "max_features": 20000}},
]


def run_one(cfg, splits):
    """Train + evaluate + log one experiment config as a single MLflow run."""
    X_train, X_test, y_train, y_test = splits
    tfidf_params = cfg.get("tfidf_params")

    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        # Per-run output dir so each run's files are kept on disk (never overwritten).
        run_dir = ARTIFACTS_DIR / f"{cfg['run_name']}_{run.info.run_id[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)

        model = build_model(estimator=cfg["estimator"], tfidf_params=tfidf_params)
        model.fit(X_train, y_train)
        metrics, cm_df, report_text = evaluate(model, X_test, y_test)

        # ── Params ──
        est = model.named_steps["clf"]
        mlflow.log_param("estimator", type(est).__name__)
        mlflow.set_tag("estimator_repr", repr(est))
        for k, v in {**DEFAULT_TFIDF_PARAMS, **(tfidf_params or {})}.items():
            mlflow.log_param(f"tfidf_{k}", v)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        # ── Eval artifacts ──
        cm_path = run_dir / "confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="eval")

        report_path = run_dir / "classification_report.txt"
        report_path.write_text(report_text)
        mlflow.log_artifact(str(report_path), artifact_path="eval")

        # ── Metrics + model ──
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, "model")

        # ── Console summary ──
        auc = f" roc_auc={metrics['roc_auc_macro']:.4f}" if "roc_auc_macro" in metrics else " (no predict_proba)"
        print(f"\n=== {cfg['run_name']} ({type(est).__name__}) ===")
        print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}{auc}")
        print(report_text)
        return metrics


def main():
    X, y, _ = load_data()
    splits = split_data(X, y)
    print(f"Loaded {len(X)} rows; train={len(splits[0])} test={len(splits[1])}")
    for cfg in EXPERIMENTS:
        run_one(cfg, splits)


if __name__ == "__main__":
    main()
