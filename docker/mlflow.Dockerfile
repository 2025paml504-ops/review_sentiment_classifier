# Lean image for the mlflow tracking server only.
#
# The server needs nothing from the ML stack (no torch/transformers/dvc), so it
# gets its own ~300 MB image instead of the full shared pipeline image - this is
# what makes `docker compose up -d mlflow` build in seconds. mlflow is pinned to
# the SAME version the trainer logs with (requirements.txt), preserving parity
# between the server and the client that writes to it.
FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.15.1

EXPOSE 5000
