"""FastAPI serving layer for the sentiment classifier (M4, added 12-Aug).

Serves the logreg model and its TF-IDF vectorizer over a REST API. logreg is
used here instead of bert_mini mainly for speed - it loads from a small
pickle in milliseconds, while bert_mini needs torch/transformers loaded just
to answer one request. See Decisions §22 for the full reasoning.

Run locally:

    uvicorn serving.app:app --reload --port 8000

Endpoints:

    GET  /health   - is it up, and which model is loaded
    POST /predict  - {"text": "..."} -> sentiment, confidence, per-class
                      probabilities, and how long the request took
"""

import logging
import time
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from features.build_features import clean_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("serving")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "model_store" / "logreg_v1.pkl"
# Duplicated from features/vectorize.py instead of imported - importing that
# module drags in feature_store.py and SQLAlchemy for one path constant this
# file doesn't otherwise need.
VECTORIZER_PATH = REPO_ROOT / "model_store" / "tfidf_vectorizer_v1.pkl"
MODEL_NAME = "logreg"

# Keeps per-request cost predictable and rejects obvious abuse.
MAX_TEXT_LENGTH = 5000

app = FastAPI(
    title="Hotel Review Sentiment Classifier",
    description="Serves the logreg model trained in training/train_linear.py",
    version="1.0",
)

# Load once at startup, not per request - reloading a pickle every call would
# cost far more than the prediction itself.
try:
    _model = joblib.load(MODEL_PATH)
    _vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError as exc:
    _model = None
    _vectorizer = None
    logger.warning(
        "Model/vectorizer not found (%s) - run `dvc repro train` first. /predict will return 503.",
        exc,
    )


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        # min_length=1 only catches a literally empty string; "   " gets past
        # that but is blank in every way that matters.
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "model": MODEL_NAME,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model is None or _vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded - run `dvc repro train` first")

    start = time.perf_counter()
    cleaned = clean_text(request.text)
    if not cleaned.strip():
        # Punctuation/numbers/placeholder-only text cleans down to nothing -
        # the same empty-document case build_features() drops during
        # training (Decisions §1). Reject it here too instead of scoring an
        # all-zero row.
        raise HTTPException(
            status_code=422,
            detail="Text cleaned to an empty document (only punctuation, numbers, "
            "or placeholder content) - nothing left to score.",
        )

    features = _vectorizer.transform([cleaned])
    proba = _model.predict_proba(features)[0]
    labels = _model.classes_
    probabilities = {label: float(p) for label, p in zip(labels, proba)}
    best_idx = int(proba.argmax())
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        sentiment=labels[best_idx],
        confidence=float(proba[best_idx]),
        probabilities=probabilities,
        latency_ms=round(latency_ms, 2),
    )
