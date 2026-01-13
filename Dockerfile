# Multi-stage build for Reflex service
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Build wheel
RUN pip install --no-cache-dir build && \
    python -m build --wheel

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 reflex && \
    mkdir -p /app/data && \
    chown -R reflex:reflex /app

# Copy wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/

# Install Python packages
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl

# Switch to app user
USER reflex

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8096/health', timeout=5.0)" || exit 1

# Run bot
CMD ["python", "-m", "reflex.services.bot"]
