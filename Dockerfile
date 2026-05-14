# Multi-stage Dockerfile for Tissue Tile Quality Filter
# Version: 0.1.4

# Stage 1: Build stage
FROM python:3.11-slim AS builder

# Install uv package manager
RUN pip install --no-cache-dir uv

WORKDIR /build

# Copy dependency files and README (required by pyproject.toml for build system)
COPY pyproject.toml README.md ./

# Copy source code
COPY src/ ./src/

# Create venv and install dependencies
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python .

# Stage 2: Runtime stage
FROM python:3.11-slim

# Install runtime dependencies (OpenCV needs some system packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source code
COPY src/ ./src/

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

LABEL org.opencontainers.image.title="Tissue Tile Quality Filter" \
      org.opencontainers.image.version="0.1.4" \
      org.opencontainers.image.description="Production-grade histology tile quality assessment pipeline"

# Set the CLI as the entrypoint
ENTRYPOINT ["python", "-m", "tissue_tile_quality_filter"]

USER appuser
