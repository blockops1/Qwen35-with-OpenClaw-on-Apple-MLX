#!/bin/bash
# start-vllm-distilled.sh — Qwen3.5 Claude Distilled (conversational, reasoning)
# Model: Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit
# Flag:  --continuous-batching (no vision weights)
# Use for: general agent work, Telegram sessions, reasoning tasks
exec vllm-mlx serve "$HOME/mlx-models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit" \
  --host 0.0.0.0 \
  --port 8091 \
  --continuous-batching \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --cache-memory-percent 0.30 \
  --kv-cache-quantization \
  --kv-cache-quantization-bits 8 \
  --chunked-prefill-tokens 1024 \
  --timeout 600
