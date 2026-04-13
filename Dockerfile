# ──────────────────────────────────────────────────────────────
#  Dockerfile — Stock Predictor FinBERT
#  Multi-stage build to keep the final image as small as possible.
#  Deploy on Render as a Web Service:
#    Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
# ──────────────────────────────────────────────────────────────

# ── Stage 1: Build dependencies ───────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed to compile some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first — leverage Docker layer cache
COPY requirements.txt .

# Install into a prefix so we can copy just the site-packages
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime image ────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home appuser

WORKDIR /app

# Pull installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appuser . .

# Create writable directories the app writes to at runtime
RUN mkdir -p data/raw data/processed models artifacts logs && \
    chown -R appuser:appuser data models artifacts logs

# Switch to non-root user
USER appuser

# ── Environment defaults (overridden by Render env vars) ──────
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose the port Render assigns via $PORT env var
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Start command ──────────────────────────────────────────────
# Render injects $PORT — we use shell form so the variable expands.
CMD uvicorn main:app --host $APP_HOST --port ${PORT:-8000} --workers 1