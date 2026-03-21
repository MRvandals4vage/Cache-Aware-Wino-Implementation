FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3-pip \
    python3-dev \
    linux-tools-common \
    linux-tools-generic \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install python requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /workspace

# Set default entrypoint
CMD ["python3", "-u", "main.py", "--platform", "jetson"]
