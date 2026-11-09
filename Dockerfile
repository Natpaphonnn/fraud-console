# Fraud console scoring API: small, non-root image. Built and tested in CI (.github/workflows/ci.yml).
#   docker build -t fraud-console-api .
#   docker run --rm -p 8000:8000 fraud-console-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/app/model

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-compile -r api/requirements.txt

COPY api/ api/
COPY reports/p4_web_model.json reports/p2_threshold.json model/

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
