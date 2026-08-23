"""Compute the reference ("what normal looks like") stats that monitor.py
compares live and simulated traffic against.

Every monitoring signal in this project is a comparison: is the current
window's confidence distribution different from this baseline, is the
current OOV rate higher than this baseline, is the current predicted class
balance different from this baseline. None of those questions are
answerable without first writing down what "normal" was - so this script
scores the held-out test split (data/processed/test_v1.csv) with the served
model and saves the result to monitoring/baseline.json.

Deliberately scores the *test* split, not the training split: the test
split is what the model's reported macro-F1/accuracy (decisions.md §22) is
actually measured on, so this baseline is directly comparable to the
numbers already trusted elsewhere in this project.

Run it once after training (or retraining) rnn_lstm:

    python -m monitoring.baseline
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("monitoring.baseline")

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = REPO_ROOT / "data" / "processed" / "test_v1.csv"
BASELINE_PATH = REPO_ROOT / "monitoring" / "baseline.json"


def oov_rate(cleaned_text: str, vocab: dict) -> tuple[int, int]:
    """(token_count, oov_token_count) for one cleaned review against a vocabulary."""
    tokens = cleaned_text.split()
    if not tokens:
        return 0, 0
    oov = sum(1 for t in tokens if t not in vocab)
    return len(tokens), oov


def build_baseline(limit: int | None = None) -> dict:
    # Imported here, not at module level: importing serving.app loads the
    # trained model into memory immediately, which a plain `import
    # monitoring.baseline` (e.g. from monitor.py, which only needs the
    # already-written JSON) shouldn't have to pay for.
    from serving.app import ID2LABEL, MAX_LENGTH, OOV_ID, _encode, _model, _vocab
    import torch
    import torch.nn.functional as F

    if _model is None or _vocab is None:
        raise RuntimeError("rnn_lstm model not loaded - run `dvc repro train_rnn` first")

    df = pd.read_csv(TEST_CSV, nrows=limit)
    df["clean_review"] = df["clean_review"].fillna("").astype(str)
    df = df[df["clean_review"].str.strip().astype(bool)].reset_index(drop=True)

    confidences, predicted, token_counts, oov_counts = [], [], [], []
    for cleaned in df["clean_review"]:
        n_tokens, n_oov = oov_rate(cleaned, _vocab)
        token_counts.append(n_tokens)
        oov_counts.append(n_oov)
        inputs, lengths = _encode(cleaned)
        with torch.no_grad():
            logits = _model(inputs, lengths)[0]
        proba = F.softmax(logits, dim=0).tolist()
        best_idx = int(logits.argmax())
        confidences.append(float(proba[best_idx]))
        predicted.append(ID2LABEL[best_idx])

    predicted = np.array(predicted)
    confidences = np.array(confidences)
    total_tokens = sum(token_counts)
    total_oov = sum(oov_counts)

    baseline = {
        "n_rows": int(len(df)),
        "mean_confidence": float(confidences.mean()),
        "confidence_scores": confidences.tolist(),
        "predicted_positive_share": float((predicted == "POSITIVE").mean()),
        "oov_rate": float(total_oov / total_tokens) if total_tokens else 0.0,
        "vocab_size": int(len(_vocab)),
        "source_csv": TEST_CSV.name,
    }
    return baseline


if __name__ == "__main__":
    baseline = build_baseline()
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    logger.info(
        "n=%d, mean confidence %.4f, predicted POSITIVE share %.4f, OOV rate %.4f",
        baseline["n_rows"], baseline["mean_confidence"],
        baseline["predicted_positive_share"], baseline["oov_rate"],
    )
    logger.info("Wrote baseline: %s", BASELINE_PATH)
