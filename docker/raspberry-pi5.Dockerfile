FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV OMP_NUM_THREADS=4

# Install basic python toolchain
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    linux-perf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Set up virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /workspace

# Compile C-ext fallback if applicable
RUN gcc -shared -o fused_winograd.so -fPIC -O3 -march=native fused_winograd.c || true

CMD ["python3", "-u", "main.py", "--platform", "pi5"]
