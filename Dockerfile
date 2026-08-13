# Serves the sentiment classifier's REST API (M4, added 12-Aug).
# Uses serving/requirements.txt instead of the root requirements.txt - it's
# lean again now that serving uses rnn_lstm (v1.4): no torch/transformers
# stack the way bert_mini needed, and it still skips mlflow/dvc/kaggle,
# which the API never touches at runtime.
FROM python:3.12-slim

WORKDIR /app

COPY serving/requirements.txt serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt

# Just what serving/app.py needs: build_features.py for clean_text(), and
# the trained RNN's weights plus its vocabulary file.
COPY features/__init__.py features/
COPY features/build_features.py features/
COPY serving/ serving/
COPY model_store/rnn_lstm_v1.pt model_store/rnn_lstm_v1.pt
COPY model_store/rnn_lstm_v1_vocab.json model_store/rnn_lstm_v1_vocab.json

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
