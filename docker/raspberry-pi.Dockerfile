# Use official Ubuntu 22.04 LTS (multi-architecture compatible, arm64 for Raspberry Pi)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install real system dependencies for building C-ext and running benchmarks
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    linux-tools-common \
    linux-tools-generic \
    git \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create and activate a venv (best practice for python docker)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install python dependencies required by scripts
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir numpy pandas scipy psutil matplotlib seaborn argparse

# Copy source code
COPY . /app

# Attempt C Extension compilation for the fused kernel.
# We silence errors on failure so it can fallback to NumPy if GCC fails for whatever architectural reason.
RUN cd /app && gcc -shared -o fused_winograd.so fused_winograd.c -fPIC -O3 -march=native || echo "C Extension compilation failed. Using Numpy fallback."

# Entrypoint is the main benchmark orchestrator
CMD ["python3", "benchmarks/run_all_benchmarks.py", "--mode", "micro"]
