"""MLflow experiment tracking for the training stages (v1.2).

A saved `.pkl` records what the model is, never how it got there. This module
adds the missing half: every training run logs its parameters, metrics, tags
and artifacts to MLflow, so a run can be compared against the others and
reproduced from its record alone.

`mlflow` is a required dependency, declared in `requirements.txt` and
imported at module level like any other - tracking is part of the pipeline,
not an optional add-on. A run that can't be tracked can't be reproduced
either, so this fails loudly instead of quietly training untracked.

The tracking URI defaults to `sqlite:///mlflow.db` at the repo root (override
with `MLFLOW_TRACKING_URI`), so no server is needed and everything works
offline - mlflow 3.15 put the old `./mlruns` file store into maintenance mode
and raises on it, so SQLite is the backend now; run artifacts live under
`mlartifacts/`. Browse both with `mlflow ui --backend-store-uri
sqlite:///mlflow.db`.

Every model family lands in one shared experiment, `review_sentiment`
(override with `MLFLOW_EXPERIMENT_NAME`), so the UI can rank the linear
baseline against the transformer directly.

Each run tags the git commit and logs `dvc.lock`, which pins the content
hash of the exact splits the model was trained on (see docs/versioning.md).
It also logs a feature-store snapshot (`feature_store/snapshot.json`:
sha256, row count, column types and label distribution of `feature_store.db`,
which is far too large to attach directly) and the feature schema
`validation/feature_column.json`, so a run can answer "which code, which
data, which columns?" entirely on its own.

`requirements.txt` names packages but pins no versions, so on its own it
can't answer "which library versions actually trained this" a month later.
Each run also logs `pip freeze` output (`environment/pip_freeze.txt`), added
in v1.2.

Usage:

    with start_run("logreg", params={...}, tags={...}, hypothesis="...") as run:
        ...
        run.log_metrics(metrics)
        run.log_artifact(path)
        run.set_conclusion("...")
"""

import contextlib
import logging
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn

logger = logging.getLogger("tracking")

# Repo root is two levels up from this file (training/tracking.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
# One local backend store for the whole project. SQLite, not the legacy
# `./mlruns` file store: mlflow >= 3.15 refuses the latter (maintenance mode).
DEFAULT_DB_PATH = REPO_ROOT / "mlflow.db"
# Artifacts (models, confusion matrices, dvc.lock) are files, so they keep a
# plain directory of their own next to the database.
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "mlartifacts"
DVC_LOCK = REPO_ROOT / "dvc.lock"
# The column contract every run was trained under.
FEATURE_SCHEMA = REPO_ROOT / "validation" / "feature_column.json"

# All runs of this project land in one experiment so the UI can rank them
# side by side, regardless of which trainer produced them.
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "review_sentiment")

# MLflow metrics must be scalars, so nested structures (per-class scores) are
# flattened and non-numeric entries (the confusion matrix) are skipped - they
# travel as artifacts instead.
_SKIP_METRIC_KEYS = {"confusion_matrix", "model"}


def tracking_uri() -> str:
    """Resolve the tracking URI, defaulting to the local SQLite backend store."""
    configured = os.environ.get("MLFLOW_TRACKING_URI")
    if configured:
        return configured
    return f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"


def artifact_location() -> str | None:
    """Where run artifacts are written; None when the URI is externally configured."""
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    DEFAULT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_ARTIFACT_DIR.as_uri()


def ensure_experiment() -> str:
    """Select the project experiment, creating it with a local artifact root."""
    location = artifact_location()
    if location and mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=location)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return EXPERIMENT_NAME


def git_commit() -> str | None:
    """Short git SHA of the working tree, or None outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def git_dirty() -> bool:
    """True when the working tree has uncommitted changes.

    Without this a run from edited-but-uncommitted code is indistinguishable
    from a clean one, and `git_commit` would be a false promise.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(out.stdout.strip())


def flatten_metrics(metrics: dict, prefix: str = "") -> dict:
    """Flatten the trainers' nested metrics dict into scalar MLflow metrics.

    `{"per_class": {"NEGATIVE": {"recall": 0.7}}}` becomes
    `{"per_class.NEGATIVE.recall": 0.7}`; non-numeric entries are dropped.
    """
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if key in _SKIP_METRIC_KEYS:
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=f"{name}."))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


class Run:
    """The active MLflow run: everything worth keeping goes through here."""

    @property
    def run_id(self) -> str:
        run = mlflow.active_run()
        return run.info.run_id if run else ""

    def log_params(self, params: dict) -> None:
        """Log the knobs that were turned; values are stringified by MLflow."""
        if params:
            mlflow.log_params({k: v for k, v in params.items() if v is not None})

    def log_metrics(self, metrics: dict) -> None:
        """Log every scalar in a (possibly nested) metrics dict."""
        mlflow.log_metrics(flatten_metrics(metrics))

    # Added (v1.2): the recurrent trainer produces a value per epoch, not
    # just a final one - a learning curve is how you tell an under-trained
    # net apart from one that's plateaued.
    def log_metric_step(self, key: str, value: float, step: int) -> None:
        """Log one point of a per-step metric (e.g. the loss at epoch `step`)."""
        mlflow.log_metric(key, float(value), step=step)

    def log_artifact(self, path, artifact_path: str | None = None) -> None:
        """Attach a file or directory produced by the run."""
        path = Path(path)
        if not path.exists():
            logger.warning("Nothing to log, %s does not exist", path)
            return
        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path or path.name)
        else:
            mlflow.log_artifact(str(path), artifact_path)

    def log_dict(self, payload: dict, filename: str) -> None:
        """Attach an in-memory dict as a JSON artifact (e.g. the confusion matrix)."""
        mlflow.log_dict(payload, filename)

    # Added (v1.2)
    def log_text(self, text: str, filename: str) -> None:
        """Attach an in-memory string as a text artifact (e.g. pip freeze output)."""
        mlflow.log_text(text, filename)

    # Added (v1.4): the prediction made before training started, and whether
    # it held up. `hypothesis` is set by start_run() itself; this is the
    # other half, called once the metrics are in.
    def set_conclusion(self, text: str) -> None:
        """Record what actually happened, next to the hypothesis logged at the start."""
        mlflow.set_tag("conclusion", text)

    def log_sklearn_model(self, model, artifact_path: str = "model") -> None:
        """Log a fitted estimator in MLflow's own (loadable) model format."""
        # `artifact_path` was renamed to `name` in mlflow 3; support both so the
        # run is logged whichever major version is installed.
        # Added (v1.4): skops (mlflow's safe-serialization backend) refuses to
        # save a few internal sklearn types by default, unless told they're
        # trusted - CalibratedClassifierCV (used for linear_svc) is one of
        # them. We trust these since we always logged a model we just
        # trained ourselves, never one loaded from elsewhere.
        trusted = [
            "sklearn.calibration._CalibratedClassifier",
            "sklearn.calibration._SigmoidCalibration",
        ]
        try:
            mlflow.sklearn.log_model(model, name=artifact_path, skops_trusted_types=trusted)
        except TypeError:
            mlflow.sklearn.log_model(
                model, artifact_path=artifact_path, skops_trusted_types=trusted
            )


def log_feature_inputs(run: "Run") -> None:
    """Attach the feature-store snapshot and the feature schema to the run.

    Both trainers read the same splits, which are built from the feature store
    under one column contract, so this belongs to every run rather than to one
    trainer.
    """
    # Imported here: `feature_store` pulls in pandas/sqlalchemy, which the
    # tracking layer itself does not need.
    from feature_store import feature_store

    snapshot = feature_store.snapshot()
    if snapshot is None:
        logger.warning("No feature store to snapshot at %s", feature_store.FEATURE_STORE_PATH)
    else:
        run.log_dict(snapshot, "feature_store/snapshot.json")
        mlflow.set_tag("feature_store_sha256", snapshot["sha256"][:12])
    run.log_artifact(FEATURE_SCHEMA, "schema")


# Added (v1.2): requirements.txt names packages but pins no versions, so two
# runs months apart could end up training under different library versions
# with nothing in the record to show it. pip freeze closes that gap per run.
def log_environment(run: "Run") -> None:
    """Snapshot exact installed package versions as a per-run text artifact."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("Could not capture pip freeze: %s", exc)
        return
    run.log_text(out.stdout, "environment/pip_freeze.txt")


@contextlib.contextmanager
def start_run(
    run_name: str,
    params: dict | None = None,
    tags: dict | None = None,
    hypothesis: str | None = None,  # Added (v1.4)
):
    """Open a tracked run in the project experiment and yield its `Run` handle.

    The git commit and the dataset reference (`dvc.lock`) are attached
    automatically, so a run always answers "which code and which data produced
    this?". `hypothesis`, if given, is logged as a tag before training starts -
    call `run.set_conclusion(...)` once the metrics are in to close the loop.
    """
    mlflow.set_tracking_uri(tracking_uri())
    ensure_experiment()

    with mlflow.start_run(run_name=run_name):
        run = Run()
        all_tags = {"model_type": run_name, **(tags or {})}
        commit = git_commit()
        if commit:
            all_tags["git_commit"] = commit
        all_tags["git_dirty"] = str(git_dirty()).lower()
        if hypothesis:
            all_tags["hypothesis"] = hypothesis
        mlflow.set_tags(all_tags)
        run.log_params(params or {})
        # dvc.lock pins the content hash of every input artifact - this is the
        # dataset reference that makes the run reproducible.
        run.log_artifact(DVC_LOCK, "dataset")
        log_feature_inputs(run)
        log_environment(run)
        logger.info("MLflow run %s (experiment %s)", run.run_id, EXPERIMENT_NAME)
        yield run
