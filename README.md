# Text Classification MLOps Case Study

An end-to-end, production-style machine-learning project for classifying incoming support tickets by intent. It covers ingestion and validation, TF-IDF and transformer experiments, reproducible training, REST serving, prediction logging, drift monitoring, and retraining triggers.

> The supplied `housing.csv` contains numeric California housing data and is not compatible with a text-classification case study. This repository therefore includes a deterministic demo support-ticket dataset generator. Replace it with a real CSV by setting `data.path`, `data.text_column`, and `data.label_column` in `params.yaml`.

## Architecture

```text
CSV -> validation/cleaning -> train/test split -> TF-IDF -> classifier -> model artifact
                                                                        |
client -> FastAPI /predict -> validation -> prediction + confidence -> JSONL logs
                                                                        |
reference metrics <---------------- drift monitor (PSI/OOV/label mix) <-+
                                      |
                               retraining decision
```

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
python scripts/make_demo_data.py
python -m text_classifier.data
python -m text_classifier.train
uvicorn text_classifier.api:app --host 0.0.0.0 --port 8000
```

Try the service:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I was charged twice for my subscription"}'
```

Interactive API docs are at `http://localhost:8000/docs`. Health and model metadata are exposed at `/health` and `/model-info`.

## Four-week workflow

### Week 1 — data and features

```bash
python scripts/make_demo_data.py
python -m text_classifier.data
dvc add data/raw/support_tickets.csv   # optional, if DVC is installed
```

Validation rejects missing columns, null/empty text, invalid labels, duplicates, and datasets too small for a stratified split. Cleaned, versioned data and a validation report are written under `data/processed/`.

### Week 2 — experiments

```bash
python -m text_classifier.train
python scripts/train_transformer.py --help
pip install mlflow  # optional UI dependencies
set MLFLOW_ALLOW_FILE_STORE=true  # Windows; use export on Linux/macOS
mlflow ui --backend-store-uri ./mlruns
```

The classical runner compares logistic regression, linear SVM, and multinomial Naive Bayes using the same split, records parameters/metrics/artifacts in MLflow, and promotes the best macro-F1 model. The transformer script is optional (`pip install -e ".[transformer]"`) because fine-tuning is substantially more expensive.

### Week 3 — REST deployment

```bash
docker build -t ticket-classifier .
docker run --rm -p 8000:8000 -v "${PWD}/logs:/app/logs" ticket-classifier
```

Requests are bounded, stripped, and validated. Empty, malformed, oversized, and control-character-only text receives a 4xx response. Responses include model version and confidence when supported.

### Week 4 — monitoring and retraining

```bash
python scripts/simulate_traffic.py --normal 100 --drifted 100
python -m text_classifier.monitor --log-path logs/predictions.jsonl
```

The monitor compares production inputs with training reference statistics using token out-of-vocabulary rate, text-length PSI, prediction-distribution Jensen-Shannon divergence, and confidence. It writes `reports/drift_report.json` and exits with code `2` when retraining thresholds are breached.

## Testing and reproducibility

```bash
pytest
ruff check .
dvc repro                 # optional orchestration
```

Random seeds, split indices, environment metadata, dataset SHA-256, configuration, metrics, and artifacts are captured. GitHub Actions runs linting, tests, demo-data generation, training, and an API smoke test.

## Repository layout

```text
src/text_classifier/      application package
scripts/                  demo, transformer, and traffic utilities
data/raw/                 immutable source data (DVC-ready)
data/processed/           validated data and reference statistics
models/                   promoted model and metadata
logs/                     append-only prediction events
reports/                  metrics and drift reports
tests/                    unit and integration tests
```

## Retraining policy

Retraining is triggered when at least two drift signals exceed thresholds, or when delayed ground-truth macro-F1 drops below the configured floor. A production workflow should require a minimum event count, validate the new dataset, retrain against a frozen evaluation set, compare against the champion, run API tests, and deploy only if quality and safety gates pass. See [`docs/retraining_strategy.md`](docs/retraining_strategy.md).

## Responsible-use notes

The demo data is synthetic. Do not log raw customer text in production without a retention policy and PII redaction. Monitor per-class and segment performance, establish a human-review path for low-confidence/high-impact tickets, and do not interpret drift as proof that model quality declined.
