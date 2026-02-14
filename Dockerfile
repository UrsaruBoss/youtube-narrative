# Dockerfile
# ----------
# GPU-ready image for your pipeline scripts (translate/sentiment/stance/tactics/report).
# Notes:
# - Use NVIDIA Container Toolkit on host.
# - Run with: docker compose run --rm pipeline python 90_report.py
# - For heavy GPU inference: use the "runtime: nvidia" compose stanza.

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (git optional, useful for some pip installs; curl for debugging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps
# If you already have requirements.txt, this will use it.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Copy project
COPY . /app

# Default command (override in compose)
CMD ["python", "--version"]
