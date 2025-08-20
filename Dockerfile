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
RUN mkdir -p /outputs /workspace/hf-cache

# Pre-fetch deps (faster subsequent builds)
RUN cargo fetch

# Build GPU binary (disable nvidia-smi check at build time)
RUN CANDLE_CUDA_DISABLE_NVSMI=1 cargo build --release

# Start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Default CMD uses env vars to run a single caption and exit
CMD ["/start.sh"]
