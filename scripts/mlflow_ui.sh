#!/usr/bin/env bash
# Launch the MLflow UI against this project's SQLite tracking store.
#
# Uses port 5001 by default because macOS AirPlay Receiver squats on port 5000
# and returns 403. Override with the first arg: ./scripts/mlflow_ui.sh 5002
#
# Usage:
#   ./scripts/mlflow_ui.sh          # http://127.0.0.1:5001
#   ./scripts/mlflow_ui.sh 5055     # custom port
set -euo pipefail

# Repo root = parent of this script's directory, regardless of where it's called from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-5001}"

# Prefer the project venv's mlflow if present, else fall back to PATH.
MLFLOW="$REPO_ROOT/.venv/bin/mlflow"
[ -x "$MLFLOW" ] || MLFLOW="mlflow"

echo "MLflow UI  ->  http://127.0.0.1:$PORT   (store: mlflow.db)"
exec "$MLFLOW" ui \
  --backend-store-uri "sqlite:///$REPO_ROOT/mlflow.db" \
  --host 127.0.0.1 \
  --port "$PORT"
