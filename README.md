# JoyCaption Rust - Local CPU Version

A Rust implementation of the JoyCaption model that runs locally on CPU, converting Python-based image captioning to efficient Rust code.

## Prerequisites

### For Native Rust Execution (Recommended for Local Development)
- Rust (latest stable version) - Install from [rustup.rs](https://rustup.rs/)
- An image file to caption (or use the included `caption-image.png`)

### For Docker Execution
- Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- Ensure Docker daemon is running and `docker` command is accessible

## Local Setup and Usage

### Option 1: Native Rust Execution (Faster, Direct)

#### Building
```bash
cargo build --release
```

#### Running
```bash
# Using the start script (recommended)
bash start.sh

# Or run directly
./target/release/joycaption-candle \
  --image ./caption-image.png \
  --prompt "Describe the image in detail." \
  --cpu
```

### Option 2: Docker Execution (Containerized)

#### Building Docker Image
```bash
docker build -t joycaption-candle .
```

#### Running with Docker
```bash
docker run --rm -v $(pwd):/app joycaption-candle
```

### Environment Variables
- `HF_HUB_TOKEN`: Optional HuggingFace token for private models
- `IMAGE_PATH`: Path to image file (default: `./caption-image.png`)
- `PROMPT`: Caption prompt (default: "Describe the image in detail.")
- `MAX_NEW_TOKENS`: Maximum tokens to generate (default: 128)
- `TEMPERATURE`: Sampling temperature (default: 0.6)
- `CPU`: Force CPU usage (default: true for local execution)

### Troubleshooting
- **"cargo: command not found"**: Install Rust from [rustup.rs](https://rustup.rs/)
- **"docker: command not found"**: Install Docker Desktop or Docker Engine
- **Model download issues**: Ensure internet connection and optionally set `HF_HUB_TOKEN`

### Notes
- First run will download the model (~2-3GB) from HuggingFace
- Model files are cached locally for subsequent runs
- CPU execution is slower than GPU but works on any machine without special hardware
- Native Rust execution is generally faster than Docker for local development
