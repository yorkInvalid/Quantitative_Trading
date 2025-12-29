# =============================================================================
# Stage 1: Base image with system dependencies (rarely changes)
# =============================================================================
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Layer 1: System dependencies (almost never changes)
# - build-essential, cmake: Required for compiling Qlib and LightGBM
# - libxml2-dev, libxslt-dev: Required for lxml (akshare dependency)
# - libgomp1: Required for LightGBM OpenMP support
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libgomp1 \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# =============================================================================
# Stage 2: Python dependencies (changes only when requirements.txt changes)
# =============================================================================
# Layer 2: Copy ONLY requirements.txt first
COPY requirements.txt /app/requirements.txt

# Layer 3: Install Python dependencies (cached unless requirements.txt changes)
# Qlib compilation is time-consuming, so this layer is critical for caching
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# =============================================================================
# Stage 3: Application code (changes frequently)
# =============================================================================
# Layer 4: Copy source code (invalidates only this layer on code changes)
COPY src/ /app/src/
COPY config/ /app/config/
COPY tests/ /app/tests/

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app

# Default command
CMD ["bash"]
