# Serving

REST API for the sentiment classifier (M4). Serves `rnn_lstm` - the highest
macro-F1 of the four trained models on the current data, and macro-F1 is the
metric this whole project has used throughout. See [Decisions §22](../docs/design/decisions.md)
for the full reasoning, including why `bert_mini` was the pick on an earlier
data snapshot and a calibration experiment on `linear_svc` that raised its
accuracy but lowered its macro-F1.

## Run it

```bash
# locally, from the repo root (needs the full requirements.txt installed)
uvicorn serving.app:app --reload --port 8000

# or via Docker
docker build -t review-sentiment-api .
docker run -p 8000:8000 review-sentiment-api
```

Either way, `model_store/rnn_lstm_v1.pt` and `model_store/rnn_lstm_v1_vocab.json`
need to already exist. Run `dvc repro train_rnn` first if they don't (see
[Pipeline](../docs/pipeline.md)).

## Endpoints

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```
```json
{"status": "ok", "model": "rnn_lstm", "model_path": "...model_store\\rnn_lstm_v1.pt"}
```

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The room was clean and the staff were incredibly friendly, best stay ever"}'
```
```json
{
  "sentiment": "POSITIVE",
  "confidence": 0.9808,
  "probabilities": {"NEGATIVE": 0.0006, "NEUTRAL": 0.0186, "POSITIVE": 0.9808},
  "latency_ms": 17.43
}
```

Negation is handled correctly - this is the bug fix from
[Decisions §10](../docs/design/decisions.md) showing up in a live prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This was not the best hotel, the room was dirty and the staff were rude"}'
# -> {"sentiment": "NEGATIVE", "confidence": 0.9866, ...}
```

## Edge cases (actually tested, not just handled in theory)

| Input | Response |
|---|---|
| `{"text": ""}` | `422` — empty string rejected by field validation |
| `{"text": "   "}` | `422` — whitespace-only rejected by a custom validator |
| `{"text": "12345 !!! ???"}` | `422` — cleans to an empty document after `clean_text()`, rejected before scoring |
| `{}` (missing field) | `422` — FastAPI/Pydantic's built-in validation |
| `{"text": 12345}` (wrong type) | `422` — built-in type validation |
| not valid JSON at all | `422` — built-in JSON parsing error |
| text over 5000 characters | `422` — `max_length` bound, keeps per-request cost predictable |

## Latency / throughput

Measured locally (`rnn_lstm`, CPU, sequential requests, no batching):
**~69 req/s, ~14.5ms/request average**. Between the two extremes already
measured on earlier versions of this API: `logreg` (~336 req/s, ~3ms/request)
and `bert_mini` (~45 req/s, ~22ms/request) - a small trained-from-scratch
recurrent net costs more than a linear model but far less than a full
transformer. See [Decisions §22](../docs/design/decisions.md) for the full
model-choice reasoning.

## UI

`ui/index.html` is a small static page over this API - a text box, an
Analyze button, and a result view with the sentiment badge, a confidence
percentage, and a probability bar per class. No framework, no build step -
see [Decisions §23](../docs/design/decisions.md) for why.

```bash
# serve it on any port other than 8000 (the API's port)
python -m http.server 8090 --directory ui
```

Then open `http://localhost:8090/index.html` with the API running
separately on port 8000. It calls `http://127.0.0.1:8000/predict` directly
from the browser - that's a different origin than the UI's own port, which
is why `serving/app.py` has CORS enabled for local development.
