#!/usr/bin/env python3
"""
detect_flag.py — Detect the correct vllm-mlx startup flag for a Qwen3.5 model.

Usage:
    python3 detect_flag.py /path/to/mlx-models/Qwen3.5-9B-Instruct-4bit
    python3 detect_flag.py /path/to/mlx-models/Qwen3.5-27B-4bit

Background:
    Qwen3_5ForConditionalGeneration appears in ALL Qwen3.5 model configs regardless
    of whether the model includes a vision tower. Do not use the config architectures
    field to decide — check the actual safetensors weights instead.

    Models with vision tower weights → --mllm     (SimpleEngine)
    Models without vision weights   → --continuous-batching  (Batched engine, faster)
"""

import json
import struct
import sys
from pathlib import Path


def detect(model_path: str) -> None:
    path = Path(model_path)

    if not path.exists():
        print(f"ERROR: Path not found: {path}")
        sys.exit(1)

    # Find the safetensors file
    safetensors = list(path.glob("*.safetensors"))
    if not safetensors:
        # Check for index file (sharded model)
        index = path / "model.safetensors.index.json"
        if index.exists():
            print("Sharded model detected — checking index for vision layer references...")
            idx = json.loads(index.read_text())
            keys = list(idx.get("weight_map", {}).keys())
            vision_keys = [k for k in keys if "vision" in k.lower()]
            _report(model_path, len(vision_keys), vision_keys[:5])
            return
        print("ERROR: No .safetensors file found in model directory")
        sys.exit(1)

    # Read the safetensors header (first 8 bytes = header length, then JSON header)
    st_file = safetensors[0]
    print(f"Model:  {path.name}")
    print(f"File:   {st_file.name}")

    with open(st_file, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))

    keys = [k for k in header.keys() if k != "__metadata__"]
    vision_keys = [k for k in keys if "vision" in k.lower()]

    _report(model_path, len(vision_keys), vision_keys[:5])


def _report(model_path, count, samples):
    print(f"Weights: {count} vision-related")
    if samples:
        print(f"Sample vision keys: {samples}")
    print()
    if count > 0:
        print("✅ Use flag:  --mllm")
        print("   Engine:    SimpleEngine (vision + text)")
    else:
        print("✅ Use flag:  --continuous-batching")
        print("   Engine:    Batched engine (text-only, faster, prefix cache works)")
    print()
    print("vllm-mlx serve command:")
    flag = "--mllm" if count > 0 else "--continuous-batching"
    print(f"  vllm-mlx serve {model_path} \\")
    print(f"    --host 0.0.0.0 --port 8091 \\")
    print(f"    {flag} \\")
    print(f"    --tool-call-parser hermes \\")
    print(f"    --enable-auto-tool-choice")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    detect(sys.argv[1])
