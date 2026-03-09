#!/bin/bash
# start-vllm-instruct.sh — Qwen3.5 Base Instruct (software development, code generation)
# Model: Qwen3.5-27B-4bit
# Flag:  --mllm (has vision weights)
# Use for: code generation, scaffolding, software development tasks
exec vllm-mlx serve "$HOME/mlx-models/Qwen3.5-27B-4bit" \
  --host 0.0.0.0 \
  --port 8091 \
  --mllm \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --cache-memory-percent 0.30 \
  --kv-cache-quantization \
  --kv-cache-quantization-bits 8 \
  --chunked-prefill-tokens 1024 \
  --timeout 600
