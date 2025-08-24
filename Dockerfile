# CPU-only base image for local development
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable

# App
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src

# Copy file from repo root into container
COPY caption-image.png /app/caption-image.png

RUN mkdir -p /outputs

# Pre-fetch deps (faster subsequent builds)
RUN cargo fetch

# Build CPU binary
RUN cargo build --release

# Start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Default CMD uses env vars to run a single caption and exit
CMD ["/start.sh"]





