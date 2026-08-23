from __future__ import annotations

import os
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import load_config, resolve_path
from .service import ModelNotReadyError, Predictor
from .text import clean_text


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        cleaned = clean_text(value)
        if not cleaned or not re.search(r"\w", cleaned, re.UNICODE):
            raise ValueError("text must contain at least one letter or number")
        return cleaned


class PredictionResponse(BaseModel):
    label: str
    confidence: float | None
    model_version: str
    request_id: str


def create_app(predictor: Predictor | None = None) -> FastAPI:
    config = load_config()
    serving = config["serving"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor is not None:
            app.state.predictor = predictor
        else:
            model_path = resolve_path(os.getenv("MODEL_PATH", serving["model_path"]))
            metadata_path = resolve_path(os.getenv("METADATA_PATH", serving["metadata_path"]))
            log_path = resolve_path(os.getenv("PREDICTION_LOG", serving["prediction_log"]))
            try:
                app.state.predictor = Predictor(model_path, metadata_path, log_path)
            except ModelNotReadyError as exc:
                app.state.predictor = None
                app.state.load_error = str(exc)
        yield

    app = FastAPI(
        title="Support Ticket Intent Classifier",
        version="0.1.0",
        description="Classifies support tickets and logs prediction telemetry.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request):
        ready = request.app.state.predictor is not None
        payload = {"status": "ok" if ready else "not_ready", "model_loaded": ready}
        if not ready:
            payload["detail"] = getattr(request.app.state, "load_error", "model unavailable")
        return payload

    @app.get("/model-info")
    def model_info(request: Request):
        loaded = request.app.state.predictor
        if loaded is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model unavailable"
            )
        return loaded.metadata

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request):
        if len(payload.text) > int(serving["max_text_length"]):
            raise HTTPException(status_code=413, detail="text exceeds maximum length")
        loaded = request.app.state.predictor
        if loaded is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model unavailable"
            )
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        try:
            return loaded.predict(payload.text, request_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return app


app = create_app()
