# MatchHubML API + dashboard.
# Build: docker compose build   Run: docker compose up -d  (see docker-compose.yml)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch first (the default Linux wheel pulls in several GB of CUDA),
# then the rest of the runtime requirements. lightgbm is only for experiments.
COPY requirements.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && grep -vE '^(torch|lightgbm)==' requirements.txt > requirements.runtime.txt \
 && pip install -r requirements.runtime.txt \
 && rm requirements.runtime.txt

RUN useradd --create-home --uid 1000 app

COPY src/ src/
COPY models/ models/
COPY matchhubml_frontend_live.html .
# Seed data only (selected by .dockerignore). The compose file mounts a named
# volume on /app/data; Docker copies these files into it the first time the
# volume is created, and from then on the volume is the source of truth.
COPY data/ data/

# chmod too: files copied from Windows/OneDrive can arrive with read-only dirs.
RUN chown -R app:app /app/data && chmod -R u+rwX /app/data
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# Behind Traefik: trust its X-Forwarded-* headers. The port is never published,
# so only containers on the minicloud-apps network can reach it.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
