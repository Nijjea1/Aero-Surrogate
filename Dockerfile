# Aero Surrogate API (src/service). The web UI (web/) is served separately.
#
#   docker build -t aero-surrogate .
#   docker build -t aero-surrogate:stub --build-arg WITH_TORCH=0 .   # stub only, no torch
#
# See src/service/DEPLOY.md. No secrets are baked in: pass them at run time
# with --env-file pointing at a host file that is never committed.
FROM python:3.13-slim

ARG WITH_TORCH=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU torch first, from the PyTorch CPU index, so pip never pulls the CUDA wheel.
COPY requirements-service.txt requirements-model.txt ./
RUN if [ "$WITH_TORCH" = "1" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu \
        && pip install -r requirements-model.txt; \
    else \
        pip install -r requirements-service.txt; \
    fi

COPY src/ ./src/

RUN useradd --create-home --uid 10001 aero \
    && mkdir -p /data/logs \
    && chown -R aero:aero /data
USER aero

ENV AERO_RUNLOG_DIR=/data/logs \
    AERO_SPEND_LEDGER=/data/spend.jsonl
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4)"]

# Exactly one worker: run logs and the rate limiter use process-local state.
# --no-proxy-headers: the client address is the TCP peer; X-Forwarded-For is
# handled by the app only when AERO_TRUST_PROXY=1 (see DEPLOY.md).
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--no-proxy-headers"]
