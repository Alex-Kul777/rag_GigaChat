# Multi-stage Dockerfile with optional GPU support
# Build: docker build --build-arg GPU_ENABLED=true .
# Or: docker build --build-arg GPU_ENABLED=false .

ARG GPU_ENABLED=false
ARG CUDA_VERSION=12.4
ARG CUDNN_VERSION=8.8

# Stage 1: CUDA base (for GPU mode)
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS cuda-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies in CUDA environment
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt

# Stage 2: CPU base (for CPU mode)
FROM python:3.11-slim AS cpu-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Final image (selective based on GPU_ENABLED)
FROM ${GPU_ENABLED,true:cuda-base,cpu-base} AS runtime

WORKDIR /app

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/cache data/vectorstore experiments logs

# Setup environment variables for CUDA (when GPU is enabled)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "src/rag_gigachat/ui/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
