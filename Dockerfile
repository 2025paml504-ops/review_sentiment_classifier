# syntax=docker/dockerfile:1
# One image, reused by every Python service in docker-compose.yml
# (mlflow, trainer, api, monitor), each differentiated by its `command`.
# A single image guarantees the mlflow the server runs matches the mlflow the
# trainer logs with, and that the serving code matches the code that trained
# the model - the main way train/serve environment skew is avoided.
FROM python:3.12-slim

# Non-root runtime user (defense in depth: a compromised process has no root).
RUN groupadd --system appuser \
 && useradd --system --gid appuser --create-home appuser

WORKDIR /app

# CPU-only torch: on Linux `pip install torch` pulls the CUDA wheel plus multi-GB
# nvidia-* libraries by default. Nothing here uses a GPU (serving loads with
# map_location="cpu"; the LSTM is tiny), so the CPU wheel keeps the image
# ~3-4 GB smaller. The --mount cache lets rebuilds reuse the downloaded wheel.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Lean runtime deps for the Dockerized services (rnn_lstm trainer, api, monitor,
# mlflow client). requirements-docker.txt deliberately drops the transformer
# stack, dvc, and kaggle - none are imported by the code these containers run -
# which is most of the build time. Cache mount speeds up rebuilds.
COPY requirements-docker.txt requirements-docker.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-docker.txt

# All pipeline + serving + monitoring code. Bulk/generated paths (data/,
# .dvc/cache, .venv, model_store/*, *.db, ...) are excluded via .dockerignore;
# data/ and the model are provided at runtime via volumes.
COPY . .

# Writable dirs for runtime artifacts/data + the mlflow artifact root, handed
# to the non-root user. chown runs after COPY (which lands root-owned files)
# so it covers the whole tree.
RUN mkdir -p /app/artifacts /app/data /mlflow/artifacts \
 && chown -R appuser:appuser /app /mlflow

USER appuser

EXPOSE 8000
# Default command = the API. trainer / monitor / mlflow override `command`.
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
