"""Run the hand-labeled drift set (monitoring/drift_samples.py) through the
served model, score it, and log every prediction - simulating what a batch
of newer, slangier customer traffic would look like hitting this API.

Every row has a hand-assigned true_sentiment, which real API traffic never
has at request time (the ground-truth-delay problem, docs/monitoring.md).
That's exactly what makes this useful: it's the one dataset in this project
where monitor.py can compute genuine accuracy/macro-F1 on demand, instead
of only inferring drift indirectly from confidence and vocabulary stats.

Run it:

    python -m monitoring.simulate_drift
"""

import json
import logging
from pathlib import Path

import numpy as np

from monitoring import prediction_log
from monitoring.baseline import BASELINE_PATH, oov_rate
from monitoring.drift_samples import DRIFT_SAMPLES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("monitoring.simulate_drift")

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "monitoring" / "drift_report.json"


def run() -> dict:
    from serving.app import ID2LABEL, MODEL_VERSION, _encode, _model, _vocab, clean_text
    import time
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, f1_score

    if _model is None or _vocab is None:
        raise RuntimeError("rnn_lstm model not loaded - run `dvc repro train_rnn` first")

    y_true, y_pred, token_counts, oov_counts = [], [], [], []
    for raw_text, true_label in DRIFT_SAMPLES:
        cleaned = clean_text(raw_text)
        n_tokens, n_oov = oov_rate(cleaned, _vocab)

        start = time.perf_counter()
        inputs, lengths = _encode(cleaned)
        with torch.no_grad():
            logits = _model(inputs, lengths)[0]
        proba = F.softmax(logits, dim=0).tolist()
        latency_ms = (time.perf_counter() - start) * 1000
        best_idx = int(logits.argmax())
        predicted = ID2LABEL[best_idx]
        probabilities = {ID2LABEL[i]: float(p) for i, p in enumerate(proba)}

        prediction_log.log_prediction(
            cleaned_text=cleaned,
            token_count=n_tokens,
            oov_token_count=n_oov,
            sentiment=predicted,
            confidence=float(proba[best_idx]),
            probabilities=probabilities,
            latency_ms=latency_ms,
            model_version=MODEL_VERSION,
            source="drift_simulation",
            true_sentiment=true_label,
        )

        y_true.append(true_label)
        y_pred.append(predicted)
        token_counts.append(n_tokens)
        oov_counts.append(n_oov)

    total_tokens = sum(token_counts)
    total_oov = sum(oov_counts)
    report = {
        "n_rows": len(DRIFT_SAMPLES),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "oov_rate": float(total_oov / total_tokens) if total_tokens else 0.0,
        "predicted_positive_share": float(np.mean([p == "POSITIVE" for p in y_pred])),
    }

    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        report["baseline_oov_rate"] = baseline["oov_rate"]
        report["baseline_predicted_positive_share"] = baseline["predicted_positive_share"]
    else:
        logger.warning("No baseline found (run `python -m monitoring.baseline` first) - skipping comparison")

    return report


if __name__ == "__main__":
    report = run()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "drift set: accuracy %.4f, macro-F1 %.4f, OOV rate %.4f (n=%d)",
        report["accuracy"], report["macro_f1"], report["oov_rate"], report["n_rows"],
    )
    if "baseline_oov_rate" in report:
        logger.info(
            "baseline (test split): OOV rate %.4f -> drift set OOV rate %.4f (+%.1fpp)",
            report["baseline_oov_rate"], report["oov_rate"],
            100 * (report["oov_rate"] - report["baseline_oov_rate"]),
        )
    logger.info("Wrote report: %s", REPORT_PATH)
