# Aero Surrogate web UI: the pages in web/ and the /api/mesh endpoint that
# builds the 3D car. The API is its own image (the repository's Dockerfile).
#
#   docker build -f web/Dockerfile -t aero-web .
#   docker run -d --name aero-web -p 127.0.0.1:5173:5173 \
#     -e AERO_WEB_API=https://api.aero.example.com -e AERO_WEB_MESH_PER_MINUTE=60 \
#     -e AERO_WEB_TRUST_PROXY=1 --restart unless-stopped aero-web
#
# See web/DEPLOY.md. Only numpy/trimesh/scipy, the geometry module and web/ go in
# (web/Dockerfile.dockerignore); no secrets, data or checkpoints.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=5173

WORKDIR /app
COPY requirements-web.txt ./
RUN pip install -r requirements-web.txt

COPY src/__init__.py src/contracts.py ./src/
COPY src/geometry/ ./src/geometry/
COPY web/ ./web/

RUN useradd --create-home --uid 10002 web
USER web
EXPOSE 5173

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5173/healthz', timeout=4)"]

CMD ["python", "web/serve.py", "--host", "0.0.0.0"]
