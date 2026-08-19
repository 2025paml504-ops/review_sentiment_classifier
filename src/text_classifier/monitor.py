from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

from .config import load_config, resolve_path
from .text import simple_tokens


def population_stability_index(reference: list[float], current: list[float]) -> float:
    epsilon = 1e-6
    expected = np.clip(np.asarray(reference, dtype=float), epsilon, None)
    actual = np.clip(np.asarray(current, dtype=float), epsilon, None)
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed prediction log at line {line_number}") from exc
    return events


def analyze(events: list[dict], reference: dict, config: dict) -> dict:
    cfg = config["monitoring"]
    if len(events) < int(cfg["min_events"]):
        return {
            "event_count": len(events),
            "status": "insufficient_data",
            "required_events": int(cfg["min_events"]),
            "retrain": False,
            "signals": {},
        }

    vocabulary = set(reference["vocabulary"])
    tokens = [token for event in events for token in simple_tokens(str(event.get("text", "")))]
    oov_rate = sum(token not in vocabulary for token in tokens) / max(len(tokens), 1)
    lengths = [len(simple_tokens(str(event.get("text", "")))) for event in events]
    counts, _ = np.histogram(lengths, bins=reference["length_bins"])
    length_distribution = (counts / max(counts.sum(), 1)).tolist()
    length_psi = population_stability_index(reference["length_distribution"], length_distribution)

    labels = sorted(
        set(reference["prediction_distribution"]) | {str(e["prediction"]) for e in events}
    )
    ref_dist = np.array([reference["prediction_distribution"].get(label, 0.0) for label in labels])
    event_counts = Counter(str(event["prediction"]) for event in events)
    cur_dist = np.array([event_counts[label] / len(events) for label in labels])
    prediction_js = float(jensenshannon(ref_dist + 1e-12, cur_dist + 1e-12) ** 2)
    confidences = [e.get("confidence") for e in events if e.get("confidence") is not None]
    low_confidence_rate = sum(
        float(value) < float(config["serving"]["low_confidence_threshold"]) for value in confidences
    ) / max(len(confidences), 1)

    values = {
        "oov_rate": oov_rate,
        "length_psi": length_psi,
        "prediction_js": prediction_js,
        "low_confidence_rate": low_confidence_rate,
    }
    thresholds = {
        "oov_rate": float(cfg["oov_rate_threshold"]),
        "length_psi": float(cfg["length_psi_threshold"]),
        "prediction_js": float(cfg["prediction_js_threshold"]),
        "low_confidence_rate": float(cfg["low_confidence_rate_threshold"]),
    }
    signals = {
        key: {
            "value": round(value, 6),
            "threshold": thresholds[key],
            "breached": value > thresholds[key],
        }
        for key, value in values.items()
    }
    breach_count = sum(item["breached"] for item in signals.values())
    retrain = breach_count >= int(cfg["signals_to_trigger"])
    return {
        "event_count": len(events),
        "status": "drift_detected" if retrain else "healthy",
        "breached_signals": breach_count,
        "retrain": retrain,
        "signals": signals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor production text and prediction drift")
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--output", default="reports/drift_report.json")
    args = parser.parse_args()
    config = load_config(args.config)
    log_path = resolve_path(args.log_path or config["serving"]["prediction_log"])
    reference = json.loads(
        resolve_path(config["monitoring"]["reference_path"]).read_text(encoding="utf-8")
    )
    report = analyze(load_events(log_path), reference, config)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["retrain"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
