# ═══════════════════════════════════════════════════════════════════
# SolQuant Inference Engine — Dockerfile
# ═══════════════════════════════════════════════════════════════════
# Multi-stage build for the Python FastAPI inference server.
# Uses NVIDIA CUDA base image for GPU-accelerated llama-cpp-python.
#
# Target GPU:  NVIDIA MX550 (2 GB GDDR6, compute capability 7.5)
# Target OS:   Linux Mint (Ubuntu-based)
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: Build llama-cpp-python with CUDA support ───────────
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create venv to isolate build dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .

# Build llama-cpp-python with CUDA backend
# CMAKE_ARGS tells llama.cpp to compile with cuBLAS/CUDA support
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV CUDA_DOCKER_ARCH="all"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Slim runtime image ────────────────────────────────
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r solquant && useradd -r -g solquant -m solquant

WORKDIR /app

# Copy application code
COPY config.py main.py model_loader.py schemas.py server.py vram_monitor.py ./
COPY rag/ ./rag/

# Create directories for model cache and logs
RUN mkdir -p /app/models /app/logs && \
    chown -R solquant:solquant /app

USER solquant

# FastAPI default port
EXPOSE 8000

# Health check against the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python3", "main.py"]
