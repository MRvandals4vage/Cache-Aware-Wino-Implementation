FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV OMP_NUM_THREADS=4
ENV ARMCL_COMMAND="bash /workspace/tools/run_armcl.sh {c_in} {c_out} {height} {width} {kernel} {runs} {warmup}"

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3-pip \
    python3-dev \
    linux-tools-common \
    linux-tools-generic \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install python requirements including ONNX, ONNXRuntime, and TVM
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt || true && \
    pip3 install --no-cache-dir onnx onnxruntime apache-tvm || true

# Copy source code
COPY . /workspace

# Compile C extension and harnesses
RUN bash scripts/setup_device_dependencies.sh || true

# Set default entrypoint
CMD ["bash", "scripts/run_jetson_nano_benchmark.sh"]
