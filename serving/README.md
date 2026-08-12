# Serving

REST API for the sentiment classifier (M4). Serves the `logreg` model, not
the higher-scoring `bert_mini` - see [Decisions §22](../docs/design/decisions.md)
for why.

## Run it

```bash
# locally, from the repo root (needs the full requirements.txt installed)
uvicorn serving.app:app --reload --port 8000

# or via Docker (lean image, only serving/requirements.txt)
docker build -t review-sentiment-api .
docker run -p 8000:8000 review-sentiment-api
```

Either way, `model_store/logreg_v1.pkl` and `model_store/tfidf_vectorizer_v1.pkl`
need to already exist. Run `dvc repro train` first if they don't (see
[Pipeline](../docs/pipeline.md)).

## Endpoints

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```
```json
{"status": "ok", "model": "logreg", "model_path": "...model_store\\logreg_v1.pkl"}
```

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The room was clean and the staff were incredibly friendly"}'
```
```json
{
  "sentiment": "POSITIVE",
  "confidence": 0.9536,
  "probabilities": {"NEGATIVE": 0.0165, "NEUTRAL": 0.0298, "POSITIVE": 0.9536},
  "latency_ms": 1.2
}
```

Negation is handled correctly - this is the bug fix from
[Decisions §10](../docs/design/decisions.md) showing up in a live prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This was not the best hotel, the room was dirty and the staff were rude"}'
# -> {"sentiment": "NEGATIVE", "confidence": 0.978, ...}
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

Measured locally (`logreg` on TF-IDF, sequential requests, no batching):
**~336 req/s, ~3ms/request average**, including HTTP overhead. `logreg` is
fast enough that request handling costs more than the model does. See
[Decisions §22](../docs/design/decisions.md) for why that mattered when
choosing which model to serve.
