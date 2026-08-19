FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md params.yaml ./
COPY src ./src
COPY models ./models
RUN pip install --no-cache-dir .
RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "text_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]

