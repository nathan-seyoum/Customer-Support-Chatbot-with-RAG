# syntax=docker/dockerfile:1.7

# ──────────────────────────── Stage 1: builder ────────────────────────────
# Install Python deps in a virtualenv we'll copy into the runtime stage.
# This avoids shipping build-essential into the final image.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only metadata first so the dep layer caches independently of source.
COPY pyproject.toml ./
COPY app ./app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip \
 && pip install .

# Optional: pre-warm HuggingFace caches so the first request isn't a 500MB download.
# Disabled by default to keep the image lean; enable with --build-arg PREWARM=1.
ARG PREWARM=0
ARG EMBEDDER_MODEL=BAAI/bge-small-en-v1.5
ARG NLI_MODEL=cross-encoder/nli-deberta-v3-base
RUN if [ "$PREWARM" = "1" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDER_MODEL}')" && \
      python -c "from sentence_transformers import CrossEncoder; CrossEncoder('${NLI_MODEL}')"; \
    fi


# ──────────────────────────── Stage 2: runtime ────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/root/.cache/huggingface

# Non-root user for the runtime container.
RUN useradd --create-home --shell /bin/bash app

WORKDIR /srv/app

# Bring over the venv from the builder.
COPY --from=builder /opt/venv /opt/venv
# Bring over any pre-warmed HF cache from the builder stage.
COPY --from=builder /root/.cache /root/.cache

# Application code, web assets, and a starter corpus.
COPY app  ./app
COPY web  ./web
COPY data/docs ./data/docs

# Persistence directory for ChromaDB (mount a volume here in compose / prod).
RUN mkdir -p /srv/app/data/chroma && chown -R app:app /srv/app /root/.cache

USER app

EXPOSE 8000

# Health-check hits the API's own /health endpoint.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
