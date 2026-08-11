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
    python -m training.compare_runs --md             # also write docs/leaderboard.md
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from training import tracking

# Added by Ankita 11-Aug: mlflow.db/mlartifacts are git-ignored (per-machine
# tracking store), so scores never reach GitHub on their own. --md renders this
# same leaderboard as a committable markdown table.
DEFAULT_MD_PATH = "docs/leaderboard.md"

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


# Added by Ankita 11-Aug: the committed leaderboard should read as "where do we
# stand today", not a full run history -- keep one row per model, its most
# recent run, so re-running a model doesn't pile up duplicate rows over time.
def latest_per_model(rows: list[dict]) -> list[dict]:
    """Keep only the most recent run for each distinct model_name."""
    best: dict[str, dict] = {}
    for row in rows:
        key = row.get("model_name") or row.get("run_name")
        current = best.get(key)
        if current is None or (row.get("started") or "") > (current.get("started") or ""):
            best[key] = row
    return list(best.values())


# Added by Ankita 11-Aug: markdown twin of format_table(), for the --md flag.
def format_markdown(rows: list[dict]) -> str:
    """Render the leaderboard as a GitHub-flavored markdown table."""
    columns = [
        ("run_name", "Run"),
        ("model_name", "Model"),
        ("macro_f1", "macro-F1"),
        ("accuracy", "Accuracy"),
        ("roc_auc_macro", "ROC-AUC (macro)"),
        ("git_commit", "Commit"),
    ]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for row in rows:
        cells = []
        for name, _ in columns:
            value = row.get(name)
            if value is None:
                text = "-"
            elif isinstance(value, float):
                text = f"{value:.4f}"
            elif name == "git_commit":
                text = str(value)[:8]
            else:
                text = str(value)
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Added by Ankita 11-Aug: one label:value block per model -- run id, accuracy,
# ROC-AUC, macro-F1 -- the compact "where do we stand" view, as opposed to
# format_table()'s full multi-column run history.
def format_checklist(rows: list[dict]) -> str:
    """Render one label:value block per row: run id, accuracy, ROC-AUC, macro-F1."""

    def fmt(row: dict, name: str) -> str:
        value = row.get(name)
        return f"{value:.4f}" if isinstance(value, float) else "-"

    blocks = []
    for row in rows:
        blocks.append(
            f"Run ID   : {row.get('run_id') or '-'}\n"
            f"Model    : {row.get('model_name') or '-'}\n"
            f"Accuracy : {fmt(row, 'accuracy')}\n"
            f"ROC-AUC  : {fmt(row, 'roc_auc_macro')}\n"
            f"F1       : {fmt(row, 'macro_f1')}"
        )
    return "\n\n".join(blocks)


# Added by Ankita 11-Aug: an interrupted run (killed before evaluate() logs
# anything) has no score at all -- drop it, one row per model (latest run),
# ranked by sort_by. Shared by the terminal --md view and the written file so
# both show exactly the same set.
def build_leaderboard(rows: list[dict], sort_by: str) -> list[dict]:
    scored = [row for row in rows if row.get(sort_by) is not None]
    latest = latest_per_model(scored)
    latest.sort(key=lambda row: row[sort_by], reverse=True)
    return latest


# Added by Ankita 11-Aug: writes the leaderboard to disk so it survives even
# though mlflow.db itself is git-ignored.
def write_markdown(rows: list[dict], path: str, sort_by: str) -> None:
    latest = build_leaderboard(rows, sort_by)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    best_line = ""
    if latest:
        best = latest[0]
        best_line = (
            f"\n**Best model: `{best['model_name']}`** "
            f"(`{sort_by}` = {best[sort_by]:.4f}).\n"
        )
    content = (
        "# Model leaderboard\n\n"
        f"Generated {generated} by `python -m training.compare_runs --md`, "
        f"ranked by `{sort_by}`, one row per model (its latest completed run). "
        "Source: MLflow (`mlflow.db`, local/git-ignored) -- re-run after training "
        f"to refresh.\n{best_line}\n"
        f"```\n{format_checklist(latest)}\n```\n"
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="include smoke (--limit) runs")
    parser.add_argument("--sort", default="macro_f1", help="metric to rank by")
    parser.add_argument("--json", action="store_true", help="print raw rows as JSON")
    parser.add_argument(
        "--md",
        nargs="?",
        const=DEFAULT_MD_PATH,
        default=None,
        metavar="PATH",
        help=f"also write the leaderboard as markdown (default: {DEFAULT_MD_PATH})",
    )
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
    elif args.md:
        # Added by Ankita 11-Aug: mirror what gets written to the .md file --
        # one checkmark block per model's latest completed run.
        leaderboard = build_leaderboard(rows, args.sort)
        print(f"Experiment: {tracking.EXPERIMENT_NAME}  (best per model, ranked by {args.sort})")
        print(format_checklist(leaderboard))
    else:
        print(f"Experiment: {tracking.EXPERIMENT_NAME}  ({len(rows)} runs, ranked by {args.sort})")
        print(format_table(rows))

    if args.md:
        write_markdown(rows, args.md, args.sort)


if __name__ == "__main__":
    main()
