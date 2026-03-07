#!/bin/bash
# start_vllm.sh — vllm-mlx server startup script
#
# Edit MODEL_PATH and VLLM_FLAG before use.
# Run: python3 scripts/detect_flag.py /path/to/model  to determine the correct flag.
#
# Usage: bash scripts/start_vllm.sh

# ── Configure these ──────────────────────────────────────────
MODEL_PATH="$HOME/mlx-models/Qwen3.5-27B-4bit"

# Use --mllm for models WITH vision weights (27B, 35B-A3B)
# Use --continuous-batching for text-only models (9B-Instruct, 9B)
# Run detect_flag.py to determine which to use.
VLLM_FLAG="--mllm"

PORT=8091

# Cache memory: adjust based on your RAM and model size
# 64GB machine + 20GB model → 0.30 (19GB cache)
# 16GB machine + 5GB model  → 0.20 (3.2GB cache)
CACHE_PERCENT=0.30

# Request timeout in seconds (default 300 is too short for cold prefills
# of large sessions; 600 gives ~10 min headroom)
TIMEOUT=600
# ─────────────────────────────────────────────────────────────

exec vllm-mlx serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  $VLLM_FLAG \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --cache-memory-percent "$CACHE_PERCENT" \
  --kv-cache-quantization \
  --kv-cache-quantization-bits 8 \
  --chunked-prefill-tokens 2048 \
  --timeout "$TIMEOUT"
