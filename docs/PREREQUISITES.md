# Prerequisites

Read this before touching any scripts. Each item here is a silent failure waiting to happen if you skip it.

---

## Hardware

| Model | Minimum RAM | Recommended |
|-------|-------------|-------------|
| Qwen3.5-9B (any variant) | 32 GB | 32 GB |
| Qwen3.5-27B (any variant) | 64 GB | 64 GB |
| Qwen3.5-35B (any variant) | 64 GB | 96 GB+ |

Apple Silicon only (M1 / M2 / M3 / M4). Intel Macs are not supported by MLX.

---

## macOS

macOS 13.5 (Ventura) or later. Tested on macOS 26.3.

---

## ⚠️ git-lfs — Install Before Downloading Models

**If you skip this step, HuggingFace will silently download 1KB pointer files instead of actual model weights.** The download appears to succeed. The model directory looks correct. It will fail at load time with a confusing error.

```bash
brew install git-lfs
git lfs install
```

Verify:
```bash
git lfs version
# Expected: git-lfs/3.x.x (...)
```

---

## Python

Python 3.11 or later. Tested on Python 3.14.

```bash
python3 --version
```

---

## Python Dependencies

```bash
pip3 install aiohttp psutil mlx-lm huggingface_hub --break-system-packages
```

Or use the included `requirements.txt`:
```bash
pip3 install -r requirements.txt --break-system-packages
```

### ⚠️ transformers — Must Be >= 5.3.0

The `qwen3_5` model architecture is not recognized by `transformers 4.x` (the current PyPI release). Install from the HuggingFace main branch:

```bash
pip3 install git+https://github.com/huggingface/transformers.git --break-system-packages
```

Verify:
```bash
python3 -c "import transformers; print(transformers.__version__)"
# Must show 5.x or later
```

---

## vllm-mlx — Use the waybarrios Fork

**Do not install from PyPI.** The PyPI release of `vllm-mlx` has broken tool call parsing — tools appear to be called but arguments are never parsed correctly.

Install the waybarrios fork:

```bash
pip3 install git+https://github.com/waybarrios/vllm-mlx.git --break-system-packages
```

Verify:
```bash
python3 -c "import vllm_mlx; print(vllm_mlx.__version__)"
# Expected: 0.2.6 or later
```

---

## Model Download

Use `huggingface-cli` to download models. Make sure `git-lfs` is installed first (see above).

```bash
# 27B Claude Distilled (recommended — conversational + reasoning)
huggingface-cli download mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit \
  --local-dir ~/mlx-models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit

# 27B Base Instruct (alternative — code generation, software dev)
huggingface-cli download mlx-community/Qwen3.5-27B-4bit \
  --local-dir ~/mlx-models/Qwen3.5-27B-4bit

# 9B Instruct (for 32GB machines)
huggingface-cli download mlx-community/Qwen3.5-9B-Instruct-4bit \
  --local-dir ~/mlx-models/Qwen3.5-9B-Instruct-4bit
```

Each download is 14–18 GB. Expect 15–30 minutes on a fast connection.

After downloading, run `detect-flag.py` to confirm the correct vllm-mlx startup flag for your model:

```bash
python3 scripts/detect-flag.py ~/mlx-models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit
```

---

## OpenClaw

This stack is built to run as the local model backend for [OpenClaw](https://openclaw.ai), an AI agent platform.

Install OpenClaw:
```bash
npm install -g openclaw
```

Then install the Lossless Context Management plugin:
```bash
openclaw plugins install @martian-engineering/lossless-claw
```

See [`docs/openclaw-config-snippets.md`](openclaw-config-snippets.md) for the relevant configuration blocks.

---

## Telegram Bot (optional — needed for OpenClaw Telegram channel)

If you're running OpenClaw with a Telegram frontend:

1. Open Telegram → search for `@BotFather`
2. Send `/newbot` → follow prompts → copy the bot token
3. Add to `~/.openclaw/.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   ```

---

## Multi-Machine Setup (optional)

If you want a second Mac to use this machine as a remote model backend:

1. On this machine: confirm the vllm-mlx server starts with `--host 0.0.0.0` (already set in the provided `start-vllm.sh`)
2. On the remote Mac, add to `openclaw.json` providers:
   ```json
   "my-remote": {
     "baseUrl": "http://<your-mac-ip>:8080/v1",
     "apiKey": "local",
     "api": "openai-completions"
   }
   ```
3. **macOS permission:** System Settings → Privacy & Security → Local Network → enable Node.js. Without this, network requests from OpenClaw are silently blocked with no error message.
