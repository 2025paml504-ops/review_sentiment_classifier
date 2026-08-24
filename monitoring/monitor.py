"""Compare a window of logged predictions against the training-time
baseline (monitoring/baseline.py) and decide whether retraining is
justified - the evidence-based check this project runs instead of
retraining on a fixed calendar schedule.

Four signals are checked, each a practical, honestly-scoped stand-in for
the four retraining triggers described in docs/monitoring.md:

    1. Output drift    - has the confidence score's average moved too far
                          from the baseline (a z-score against the
                          baseline's mean and standard deviation)?
    2. Input drift      - has the out-of-vocabulary rate risen (a proxy for
                          "customers are using words the model has never
                          seen" - exactly what new slang/topics looks like)?
    3. Output drift      - has the predicted POSITIVE/NEGATIVE balance moved?
    4. Performance drop - *only* checkable when true labels exist for the
                          window (real traffic never has them at request
                          time - the ground-truth-delay problem). The drift
                          simulation is the one source in this project that
                          does carry true labels, so run it first to get a
                          meaningful answer to this one.

None of the four thresholds below are arbitrary - each is explained inline,
and the same reasoning is written out in full in docs/monitoring.md so a
reader doesn't have to reverse-engineer it from the code.

Run it:

    python -m monitoring.monitor --source drift_simulation   # the simulated set
    python -m monitoring.monitor --source api                # real logged traffic
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from monitoring import prediction_log
from monitoring.baseline import BASELINE_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("monitoring.monitor")

# --- Thresholds, each picked and justified independently -------------------

# |window mean confidence - baseline mean confidence| / baseline std > 1.0.
# A window whose average confidence sits more than one baseline standard
# deviation away from training-time behavior is treated as drifted - the
# same per-feature z-score threshold used for numeric-feature drift checks
# generally.
CONFIDENCE_ZSCORE_ALERT = 1.0

# +5 percentage points of out-of-vocabulary tokens above baseline. The
# training vocabulary (train_rnn.py's build_vocabulary()) covers the 20,000
# most common training-era words; a sustained rise here is the direct,
# measurable signature of "customers are using words the model has never
# seen" - new slang or new topics, concretely.
OOV_RATE_ALERT_DELTA = 0.05

# +/-10 percentage points on the share of predictions called POSITIVE. The
# training data itself is 86.5% POSITIVE (decisions.md §3); a swing this
# large means the model's output shape has changed regardless of whether
# any single input-level signal explains why.
POSITIVE_SHARE_ALERT_DELTA = 0.10

# A macro-F1 drop of 0.05 or more below the model's own reported test-set
# score (decisions.md §22: 0.8918) is treated as confirmed degradation, not
# noise - only checkable when the window has true labels attached.
MACRO_F1_DROP_THRESHOLD = 0.05
TRAINED_MACRO_F1 = 0.8918

# Below this many predictions, any of the checks above can fire from pure
# sample noise (e.g. 1 POSITIVE and 1 NEGATIVE out of 2 requests already
# swings the predicted-class share by 33 points) - real enough to hit
# during manual testing of this exact script. A window this small still
# gets scored (so the report is never silently empty), but triggers are
# downgraded to a warning rather than a retrain recommendation.
MIN_WINDOW_SIZE = 30


def check_confidence_drift(window_confidence: np.ndarray, baseline_confidence: np.ndarray) -> dict:
    baseline_mean = float(baseline_confidence.mean())
    baseline_std = float(baseline_confidence.std())
    window_mean = float(window_confidence.mean())
    z_score = abs(window_mean - baseline_mean) / baseline_std if baseline_std else 0.0
    return {
        "window_mean_confidence": round(window_mean, 4),
        "baseline_mean_confidence": round(baseline_mean, 4),
        "baseline_std_confidence": round(baseline_std, 4),
        "z_score": round(z_score, 4),
        "triggered": bool(z_score > CONFIDENCE_ZSCORE_ALERT),
    }


def check_oov_drift(window_df, baseline_oov_rate: float) -> dict:
    total_tokens = int(window_df["token_count"].sum())
    total_oov = int(window_df["oov_token_count"].sum())
    window_rate = (total_oov / total_tokens) if total_tokens else 0.0
    delta = window_rate - baseline_oov_rate
    return {
        "window_oov_rate": round(window_rate, 4),
        "baseline_oov_rate": round(baseline_oov_rate, 4),
        "delta": round(delta, 4),
        "triggered": bool(delta >= OOV_RATE_ALERT_DELTA),
    }


def check_output_balance_drift(window_df, baseline_positive_share: float) -> dict:
    window_share = float((window_df["sentiment"] == "POSITIVE").mean())
    delta = window_share - baseline_positive_share
    return {
        "window_positive_share": round(window_share, 4),
        "baseline_positive_share": round(baseline_positive_share, 4),
        "delta": round(delta, 4),
        "triggered": bool(abs(delta) >= POSITIVE_SHARE_ALERT_DELTA),
    }


def check_performance_drop(window_df) -> dict | None:
    """Only computable when the window carries true_sentiment - see module docstring."""
    labeled = window_df.dropna(subset=["true_sentiment"])
    if labeled.empty:
        return None

    from sklearn.metrics import accuracy_score, f1_score

    macro_f1 = float(f1_score(labeled["true_sentiment"], labeled["sentiment"], average="macro", zero_division=0))
    accuracy = float(accuracy_score(labeled["true_sentiment"], labeled["sentiment"]))
    drop = TRAINED_MACRO_F1 - macro_f1
    return {
        "n_labeled": int(len(labeled)),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "trained_macro_f1": TRAINED_MACRO_F1,
        "drop": round(drop, 4),
        "triggered": bool(drop >= MACRO_F1_DROP_THRESHOLD),
    }


def run(source: str = "api", limit: int | None = None) -> dict:
    if not BASELINE_PATH.exists():
        raise RuntimeError("No baseline found - run `python -m monitoring.baseline` first")
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_confidence = np.array(baseline["confidence_scores"])

    window_df = prediction_log.read_predictions(source=source, limit=limit)
    if window_df.empty:
        raise RuntimeError(
            f"No logged predictions with source={source!r} - "
            "run `python -m monitoring.simulate_drift` or send some /predict requests first"
        )

    checks = {
        "confidence_drift": check_confidence_drift(window_df["confidence"].to_numpy(), baseline_confidence),
        "oov_drift": check_oov_drift(window_df, baseline["oov_rate"]),
        "output_balance_drift": check_output_balance_drift(window_df, baseline["predicted_positive_share"]),
        "performance_drop": check_performance_drop(window_df),
    }
    triggered = [name for name, result in checks.items() if result and result.get("triggered")]
    n_predictions = int(len(window_df))
    below_min_window = n_predictions < MIN_WINDOW_SIZE

    return {
        "source": source,
        "n_predictions": n_predictions,
        "below_min_window": below_min_window,
        "checks": checks,
        # A trigger from a too-small window is real in the data but not
        # trustworthy - surfaced as a warning, not a retrain recommendation.
        "retrain_recommended": bool(triggered) and not below_min_window,
        "triggered_by": [] if below_min_window else triggered,
        "low_confidence_triggers": triggered if below_min_window else [],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="api", choices=["api", "drift_simulation"])
    parser.add_argument("--limit", type=int, default=None, help="only look at the N most recent predictions")
    args = parser.parse_args()

    result = run(source=args.source, limit=args.limit)
    print(json.dumps(result, indent=2))

    logger.info("=" * 60)
    if result["below_min_window"]:
        logger.warning(
            "Only %d predictions in this window (minimum %d) - %s not trustworthy enough to act on yet.",
            result["n_predictions"], MIN_WINDOW_SIZE,
            f"would have triggered on {', '.join(result['low_confidence_triggers'])}"
            if result["low_confidence_triggers"] else "no checks fired, and this",
        )
    elif result["retrain_recommended"]:
        logger.info("RETRAIN RECOMMENDED - triggered by: %s", ", ".join(result["triggered_by"]))
    else:
        logger.info("No retraining trigger fired - monitored traffic looks consistent with the baseline.")
