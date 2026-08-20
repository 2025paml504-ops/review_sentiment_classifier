"""FastAPI serving layer for the sentiment classifier (M4).

Serves `rnn_lstm` - the model with the highest macro-F1 (this project's
headline metric, decisions.md #14) on the current data. See decisions.md
§22 for the full history of how the served model was chosen, and §23 for
why the UI needs CORS enabled here.

How this file is organized, top to bottom:

    - Config             - which model, where its files live, request limits
    - Request/response contracts - the Pydantic contracts: what a caller
      must send, and exactly what they get back
    - Model              - the network architecture (must match
      training/train_rnn.py exactly) and loading it once at startup
    - Text -> numbers    - turning a request's raw text into what the
      model actually expects
    - Endpoints          - /health and /predict, the two things this API
      actually does

Run it:

    uvicorn serving.app:app --reload --port 8000

Then:

    GET  /health   - is it up, and which model/version is loaded
    POST /predict  - {"text": "..."} -> sentiment, confidence, per-class
                      probabilities, and how long the request took
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from features.build_features import SENTIMENT_LABELS, clean_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("serving")


# ── Config ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "rnn_lstm"
# Which artifact is actually serving, returned on every response so a
# caller can tell two deployments apart without reading logs. Update this
# whenever the served artifact changes (decisions.md §22).
MODEL_VERSION = "rnn_lstm_v1"
MODEL_PATH = REPO_ROOT / "model_store" / "rnn_lstm_v1.pt"
VOCAB_PATH = REPO_ROOT / "model_store" / "rnn_lstm_v1_vocab.json"

# These must match training/train_rnn.py exactly - they describe the shape
# of the saved weights, not a preference. Duplicated here rather than
# imported so serving doesn't pull in train_rnn.py's own dependencies
# (feature_store, SQLAlchemy) that it never actually needs at inference time.
PAD_ID = 0
OOV_ID = 1
MAX_LENGTH = 200
EMBEDDING_DIM = 128
RECURRENT_UNITS = 128
DROPOUT = 0.2

ID2LABEL = {i: label for i, label in enumerate(SENTIMENT_LABELS)}

# Keeps per-request cost predictable and rejects obvious abuse.
MAX_TEXT_LENGTH = 5000

app = FastAPI(
    title="Hotel Review Sentiment Classifier",
    description="Serves the rnn_lstm model trained in training/train_rnn.py",
    version="2.0",
)
# Open to any origin - this is a local dev API with no auth, and the static
# page in ui/ calls it from a different local port, which browsers treat
# as a different origin (decisions.md §23).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request/response contracts ───────────────────────────────────────
# These are promises: exactly what a caller must send, and exactly what
# they get back. FastAPI checks every request against PredictRequest before
# predict() ever runs - a bad request never reaches the model at all.

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
    # ge/le constrains the *response* too, not just request inputs - if a
    # future model swap ever returned something out of [0, 1] (a raw logit,
    # say), Pydantic rejects the response itself with a loud 500 instead of
    # silently handing a caller a nonsensical confidence value.
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    latency_ms: float
    model_version: str


# ── Model: architecture + loading ────────────────────────────────────

class RecurrentClassifier(nn.Module):
    """Embedding -> LSTM -> dropout -> logits. Same architecture as train_rnn.py's
    build_model("rnn_lstm", ...) - the state_dict below only loads if this matches."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=PAD_ID)
        self.rnn = nn.LSTM(EMBEDDING_DIM, RECURRENT_UNITS, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.output = nn.Linear(RECURRENT_UNITS, len(SENTIMENT_LABELS))

    def forward(self, inputs, lengths):
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.rnn(packed)
        return self.output(self.dropout(hidden[0]))


# Load once at startup, not per request - loading these fresh on every call
# would cost far more than the prediction itself. eval() turns off dropout,
# which should only run in training.
try:
    _checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    _vocab: dict[str, int] = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    _model = RecurrentClassifier(_checkpoint["vocab_size"])
    _model.load_state_dict(_checkpoint["state_dict"])
    _model.eval()
except (OSError, FileNotFoundError) as exc:
    _model = None
    _vocab = None
    logger.warning(
        "Model not found (%s) - run `dvc repro train_rnn` first. /predict will return 503.",
        exc,
    )


# ── Text -> numbers ───────────────────────────────────────────────────

def _encode(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Text -> a single padded id sequence and its real (non-padding) length.

    Same word-index lookup train_rnn.py's encode() uses: unknown words map to
    OOV_ID, and only the first MAX_LENGTH tokens are kept.
    """
    tokens = text.split()[:MAX_LENGTH]
    ids = [_vocab.get(word, OOV_ID) for word in tokens]
    encoded = np.full((1, MAX_LENGTH), PAD_ID, dtype="int64")
    if ids:
        encoded[0, : len(ids)] = ids
    length = max(len(ids), 1)
    return torch.from_numpy(encoded), torch.tensor([length])


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model is None or _vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded - run `dvc repro train_rnn` first")

    start = time.perf_counter()

    # Step 1: clean the text the same way training data was cleaned.
    cleaned = clean_text(request.text)
    if not cleaned.strip():
        # Punctuation/numbers/placeholder-only text cleans down to nothing -
        # the same empty-document case build_features() drops during
        # training (decisions.md §1). Reject it here too instead of scoring
        # against an empty string.
        raise HTTPException(
            status_code=422,
            detail="Text cleaned to an empty document (only punctuation, numbers, "
            "or placeholder content) - nothing left to score.",
        )

    # Step 2: text -> ids -> the model -> a probability per class.
    inputs, lengths = _encode(cleaned)
    with torch.no_grad():
        logits = _model(inputs, lengths)[0]
    proba = F.softmax(logits, dim=0).tolist()
    probabilities = {ID2LABEL[i]: float(p) for i, p in enumerate(proba)}
    best_idx = int(logits.argmax())
    latency_ms = (time.perf_counter() - start) * 1000

    # Step 3: package the response.
    return PredictResponse(
        sentiment=ID2LABEL[best_idx],
        confidence=float(proba[best_idx]),
        probabilities=probabilities,
        latency_ms=round(latency_ms, 2),
        model_version=MODEL_VERSION,
    )
