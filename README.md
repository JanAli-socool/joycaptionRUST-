# JoyCaption Rust - Local CPU Version

A Rust implementation of the JoyCaption model that runs locally on CPU, converting Python-based image captioning to efficient Rust code.

## Local Setup and Usage

### Prerequisites
- Rust (latest stable version)
- An image file to caption (or use the included `caption-image.png`)

### Building
```bash
cargo build --release
```

### Running
```bash
# Using the start script (recommended)
bash start.sh

# Or run directly
./target/release/joycaption-candle \
  --image ./caption-image.png \
  --prompt "Describe the image in detail." \
  --cpu
```

### Environment Variables
- `HF_HUB_TOKEN`: Optional HuggingFace token for private models
- `IMAGE_PATH`: Path to image file (default: `./caption-image.png`)
- `PROMPT`: Caption prompt (default: "Describe the image in detail.")
- `MAX_NEW_TOKENS`: Maximum tokens to generate (default: 128)
- `TEMPERATURE`: Sampling temperature (default: 0.6)
- `CPU`: Force CPU usage (default: true for local execution)

### Notes
- First run will download the model (~2-3GB) from HuggingFace
- Model files are cached locally for subsequent runs
- CPU execution is slower but works on any machine without GPU requirements
