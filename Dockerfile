# syntax=docker/dockerfile:1.7
FROM python:3.14.7-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GELATO_MODEL_PATH=/app/models/model.joblib \
    GELATO_METADATA_PATH=/app/models/metadata.json

WORKDIR /app

RUN addgroup --system --gid 10001 appgroup \
    && adduser --system --uid 10001 --ingroup appgroup --home /nonexistent appuser

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

COPY --chown=appuser:appgroup models ./models

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import json,urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); raise SystemExit(0 if data.get('model_ready') else 1)"]

CMD ["uvicorn", "gelato_forecast.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
