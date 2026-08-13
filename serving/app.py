"""FastAPI serving layer for the sentiment classifier (M4, added 12-Aug).

Serves `rnn_lstm`, not `bert_mini`. Both were candidates on macro-F1 (this
project's headline metric since Decisions §14, chosen because it doesn't let
a model coast by mostly guessing the majority POSITIVE class), but which one
actually wins changed after a data-cleaning bug fix (Decisions §10-12): on
the older, buggier data bert_mini scored higher (0.6685); on the current,
corrected data, rnn_lstm's kept 3-epoch config scores higher (0.6488 vs
0.6461) and that result reproduces exactly on repeat runs. See Decisions §22
for the full history, including the calibration experiment on `linear_svc`
that raised its accuracy but lowered its macro-F1 - the same tradeoff this
project keeps running into.

rnn_lstm also happens to be cheaper to serve than bert_mini would have been:
no `transformers` dependency, just `torch` and a small vocabulary file.

Run locally:

    uvicorn serving.app:app --reload --port 8000

Endpoints:

    GET  /health   - is it up, and which model is loaded
    POST /predict  - {"text": "..."} -> sentiment, confidence, per-class
                      probabilities, and how long the request took

CORS is open to any origin - this is a local dev API with no auth, and the
static page in ui/ needs to call it from a plain file:// or a different
localhost port, both of which browsers treat as cross-origin (Decisions §23).
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

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "rnn_lstm"
MODEL_PATH = REPO_ROOT / "model_store" / "rnn_lstm_v1.pt"
VOCAB_PATH = REPO_ROOT / "model_store" / "rnn_lstm_v1_vocab.json"

# These five constants and the RecurrentClassifier class below must match
# training/train_rnn.py exactly - they describe the shape of the saved
# weights, not a preference. Duplicated here rather than imported so serving
# doesn't pull in train_rnn.py's own dependencies (feature_store, SQLAlchemy)
# that it never actually needs at inference time.
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model is None or _vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded - run `dvc repro train_rnn` first")

    start = time.perf_counter()
    cleaned = clean_text(request.text)
    if not cleaned.strip():
        # Punctuation/numbers/placeholder-only text cleans down to nothing -
        # the same empty-document case build_features() drops during
        # training (Decisions §1). Reject it here too instead of scoring
        # against an empty string.
        raise HTTPException(
            status_code=422,
            detail="Text cleaned to an empty document (only punctuation, numbers, "
            "or placeholder content) - nothing left to score.",
        )

    inputs, lengths = _encode(cleaned)
    with torch.no_grad():
        logits = _model(inputs, lengths)[0]
    proba = F.softmax(logits, dim=0).tolist()
    probabilities = {ID2LABEL[i]: float(p) for i, p in enumerate(proba)}
    best_idx = int(logits.argmax())
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        sentiment=ID2LABEL[best_idx],
        confidence=float(proba[best_idx]),
        probabilities=probabilities,
        latency_ms=round(latency_ms, 2),
    )
