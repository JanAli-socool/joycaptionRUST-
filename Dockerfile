# CUDA runtime base (fits RunPod GPUs)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:$PATH \
    HF_HOME=/workspace/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git build-essential pkg-config libssl-dev clang \
    && rm -rf /var/lib/apt/lists/*

# Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable

# App
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src

# Copy file from repo root into container
COPY caption-image.jpg /app/caption-image.png

RUN mkdir -p /outputs /workspace/hf-cache

# Pre-fetch deps (faster subsequent builds)
RUN cargo fetch

# Skip nvcc requirement, only use runtime kernels
ENV CUDARC_DISABLE_NVCC=1

# Build GPU binary (disable nvidia-smi check at build time)
# Force skip nvidia-smi + set compute capability (Ada Lovelace = 8.9 for RTX 5090)
ENV CUDA_COMPUTE_CAP=89
RUN CANDLE_CUDA_DISABLE_NVSMI=1 CUDA_COMPUTE_CAP=$CUDA_COMPUTE_CAP cargo build --release

# Start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Default CMD uses env vars to run a single caption and exit
CMD ["/start.sh"]



