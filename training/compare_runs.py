"""Side-by-side comparison of the tracked runs (added 10 Aug - Ankita).

The MLflow UI does this interactively
(`mlflow ui --backend-store-uri sqlite:///mlflow.db`), but a leaderboard is worth
having in the terminal too: it is what you paste into a review, and it is what
proves the experiment record is actually populated without starting a server.

Reads the same store as `training/tracking.py` and prints one row per run of the
`review_sentiment` experiment, ranked by macro-F1: run id, model, the headline
metrics and the reproducibility handles (git commit, feature-store hash). Smoke
runs (`--limit`) are excluded by default -- they are not scored versions.

    python -m training.compare_runs                  # scored runs, best first
    python -m training.compare_runs --all            # include smoke runs
    python -m training.compare_runs --sort accuracy
    python -m training.compare_runs --json           # machine-readable
"""

import argparse
import json

import mlflow

from training import tracking

# The columns that make two runs comparable at a glance. Metric names are shared
# by both trainers on purpose (see training/train_transformer.py).
METRIC_COLUMNS = ["macro_f1", "accuracy", "roc_auc_macro", "weighted_f1"]
PARAM_COLUMNS = ["model_name", "random_state", "limit"]
TAG_COLUMNS = ["stage", "framework", "git_commit", "git_dirty", "feature_store_sha256"]


def fetch_runs(include_smoke: bool = False, sort_by: str = "macro_f1") -> list[dict]:
    """Flatten the experiment's runs into plain dicts, best first."""
    mlflow.set_tracking_uri(tracking.tracking_uri())
    experiment = mlflow.get_experiment_by_name(tracking.EXPERIMENT_NAME)
    if experiment is None:
        return []

    filter_string = None if include_smoke else "tags.smoke_test = 'False'"
    frame = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=[f"metrics.{sort_by} DESC"],
    )
    if frame.empty:
        return []

    rows = []
    for _, run in frame.iterrows():
        row = {
            "run_id": run["run_id"],
            "experiment": tracking.EXPERIMENT_NAME,
            "run_name": run.get("tags.mlflow.runName"),
            "started": str(run["start_time"]),
        }
        for column, prefix in (
            (PARAM_COLUMNS, "params."),
            (TAG_COLUMNS, "tags."),
            (METRIC_COLUMNS, "metrics."),
        ):
            for name in column:
                value = run.get(f"{prefix}{name}")
                row[name] = None if value != value else value  # NaN -> None
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    """Render the leaderboard as a fixed-width table."""
    columns = [
        ("run_id", 10),
        ("run_name", 12),
        ("model_name", 34),
        ("macro_f1", 9),
        ("accuracy", 9),
        ("roc_auc_macro", 13),
        ("git_commit", 11),
        ("feature_store_sha256", 20),
    ]
    header = "  ".join(name.ljust(width) for name, width in columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for name, width in columns:
            value = row.get(name)
            if value is None:
                text = "-"
            elif isinstance(value, float):
                text = f"{value:.4f}"
            elif name == "run_id":
                text = str(value)[:8]
            else:
                text = str(value)
            cells.append(text[:width].ljust(width))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="include smoke (--limit) runs")
    parser.add_argument("--sort", default="macro_f1", help="metric to rank by")
    parser.add_argument("--json", action="store_true", help="print raw rows as JSON")
    args = parser.parse_args()

    rows = fetch_runs(include_smoke=args.all, sort_by=args.sort)
    if not rows:
        print(
            f"No runs in experiment '{tracking.EXPERIMENT_NAME}' at {tracking.tracking_uri()}"
            " - train something first (python -m training.train_linear)."
        )
        return

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        print(f"Experiment: {tracking.EXPERIMENT_NAME}  ({len(rows)} runs, ranked by {args.sort})")
        print(format_table(rows))


if __name__ == "__main__":
    main()
