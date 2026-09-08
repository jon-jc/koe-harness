# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the koe service.
#
# Three things this is careful about:
#
#   1. The web client is built in a Node stage and only its output is copied
#      forward. Shipping a Node toolchain in a Python runtime image adds
#      ~200MB and a second package ecosystem's worth of CVEs to patch, for an
#      artifact that is three files.
#
#   2. Dependencies install in their own layer, before the source is copied.
#      Source changes on every commit; dependencies change rarely, so this is
#      the difference between a 10-second and a 3-minute rebuild.
#
#   3. It runs as a non-root user with a read-only-friendly layout. A voice
#      service accepts arbitrary bytes from the internet on a long-lived
#      socket, which is not the place to also be running as root.

# --------------------------------------------------------------------------
# web client
# --------------------------------------------------------------------------
FROM node:22-slim AS web

WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/tsconfig.json ./
COPY web/src ./src
RUN npx tsc --noEmit && npm run build


# --------------------------------------------------------------------------
# python dependencies
# --------------------------------------------------------------------------
FROM python:3.12-slim AS deps

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# fugashi builds a small C extension; the toolchain is needed here and
# deliberately not in the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency layer: only the metadata, so this cache survives source edits.
COPY pyproject.toml README.md ./
RUN mkdir -p src/koe && touch src/koe/__init__.py \
    # No aws extra: nothing in the service imports boto3. Metrics reach
    # CloudWatch as EMF on stdout, and secrets are injected by ECS at task
    # start, so the SDK would be ~100MB of unused attack surface.
    && pip install --no-cache-dir ".[api,llm,ja,cli]"


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    KOE_ENVIRONMENT=production \
    KOE_STRUCTURED_LOGS=true

# curl is for the container healthcheck; nothing else is added.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 koe

COPY --from=deps /opt/venv /opt/venv

WORKDIR /app
COPY --chown=koe:koe src/ ./src/
COPY --chown=koe:koe pyproject.toml README.md ./
COPY --chown=koe:koe datasets/ ./datasets/
COPY --chown=koe:koe web/index.html ./web/index.html
COPY --from=web --chown=koe:koe /build/dist ./web/dist

# Install the package itself without touching the dependency layer.
RUN pip install --no-cache-dir --no-deps -e . \
    && chown -R koe:koe /app

USER koe
EXPOSE 8000

# Readiness, not liveness: a saturated instance should be pulled from the
# load balancer without being restarted.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# One worker per container. Uvicorn workers do not share the in-process
# session state, ledger or circuit breakers, so scaling is horizontal —
# more tasks, not more workers — and each container's metrics describe
# exactly what it served.
CMD ["uvicorn", "koe.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-graceful-shutdown", "20", \
     "--no-access-log"]
