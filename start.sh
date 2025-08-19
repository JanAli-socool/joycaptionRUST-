#!/usr/bin/env bash
set -euo pipefail

# Env (set these in RunPod "Container Environment Variables")
MODEL_ID="${MODEL_ID:-fancyfeast/llama-joycaption-beta-one-hf-llava}"
IMAGE_PATH="${IMAGE_PATH:-/app/example.jpg}"
PROMPT="${PROMPT:-Describe the image in detail.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.6}"
CPU="${CPU:-false}"

echo "Model: $MODEL_ID"
echo "Image: $IMAGE_PATH"
echo "Prompt: $PROMPT"
echo "Max new tokens: $MAX_NEW_TOKENS, Temperature: $TEMPERATURE, CPU: $CPU"

# If your HF repo is gated/private, supply HF_TOKEN in RunPod env
# export HF_TOKEN=hf_xxx   # optional

/app/target/release/joycaption-candle \
  --model-id "$MODEL_ID" \
  --image "$IMAGE_PATH" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  $( [ "$CPU" = "true" ] && echo --cpu )

echo "Done."
