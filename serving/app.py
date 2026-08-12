"""FastAPI serving layer for the sentiment classifier (M4, added 12-Aug).

Serves the default `logreg` model + its fitted TF-IDF vectorizer behind a REST
API. `logreg` is used here, not `bert_mini`, because serving needs to load and
respond fast: `logreg` loads from a small pickle in milliseconds with no heavy
ML runtime, while `bert_mini` needs `torch`/`transformers` resident in memory
just to answer one request - see Decisions §22 for the full tradeoff.

Run locally:

    uvicorn serving.app:app --reload --port 8000

Endpoints:

    GET  /health   - liveness check + which model is loaded
    POST /predict  - {"text": "..."} -> sentiment, confidence, per-class
                      probabilities, and the request's own latency
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
# Path duplicated from features/vectorize.py rather than imported: importing
# that module pulls in feature_store.py -> SQLAlchemy transitively, for one
# path constant serving doesn't otherwise need. Keeps the serving image's
# real dependency chain to pandas/numpy/scikit-learn/joblib only.
VECTORIZER_PATH = REPO_ROOT / "model_store" / "tfidf_vectorizer_v1.pkl"
MODEL_NAME = "logreg"

# Bounds vectorizer/inference cost per request and rejects obvious abuse
# before it ever reaches the model.
MAX_TEXT_LENGTH = 5000

app = FastAPI(
    title="Hotel Review Sentiment Classifier",
    description="Serves the logreg model trained in training/train_linear.py",
    version="1.0",
)

# Loaded once at import time, not per-request: reloading a pickle on every
# call would dominate latency far more than the actual prediction does.
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
        # min_length=1 only rejects a literally empty string; "   " passes
        # that check but is blank in every way that matters.
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
        # Text that is only punctuation/numbers/placeholder phrases cleans
        # down to nothing - the same "empty document" case build_features()
        # drops at training time (Decisions §1). Serving rejects it too
        # rather than silently scoring an all-zero feature row.
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
