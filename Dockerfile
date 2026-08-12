# Serves the sentiment classifier's REST API (M4, added 12-Aug).
# Builds a lean image via serving/requirements.txt, not the root
# requirements.txt - see that file for why (avoids ~2GB of training-only
# torch/transformers this endpoint never imports).
FROM python:3.12-slim

WORKDIR /app

COPY serving/requirements.txt serving/requirements.txt
RUN pip install --no-cache-dir -r serving/requirements.txt

# Only what serving/app.py actually imports: features/build_features.py for
# clean_text(), and the trained model + vectorizer it serves.
COPY features/__init__.py features/
COPY features/build_features.py features/
COPY serving/ serving/
COPY model_store/logreg_v1.pkl model_store/tfidf_vectorizer_v1.pkl model_store/

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
