from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .text import clean_text


class ModelNotReadyError(RuntimeError):
    pass


class Predictor:
    def __init__(self, model_path: Path, metadata_path: Path, log_path: Path):
        if not model_path.exists() or not metadata_path.exists():
            raise ModelNotReadyError(
                "Model artifacts are missing. Run python -m text_classifier.train"
            )
        self.model = joblib.load(model_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _confidence(self, text: str) -> float | None:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([text])[0]
            return float(np.max(probabilities))
        if hasattr(self.model, "decision_function"):
            scores = np.atleast_1d(self.model.decision_function([text])[0]).astype(float)
            scores -= scores.max()
            probabilities = np.exp(scores) / np.exp(scores).sum()
            return float(probabilities.max())
        return None

    def predict(self, raw_text: str, request_id: str) -> dict[str, Any]:
        text = clean_text(raw_text)
        label = str(self.model.predict([text])[0])
        confidence = self._confidence(text)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "text": text,
            "text_length": len(text),
            "prediction": label,
            "confidence": confidence,
            "model_version": self.metadata["model_version"],
        }
        line = json.dumps(event, ensure_ascii=False, allow_nan=False)
        with self._lock, self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return {
            "label": label,
            "confidence": None
            if confidence is None or math.isnan(confidence)
            else round(confidence, 6),
            "model_version": self.metadata["model_version"],
            "request_id": request_id,
        }
