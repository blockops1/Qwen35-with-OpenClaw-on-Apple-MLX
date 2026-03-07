# Qwen3.5 with OpenClaw on Apple MLX

**Tool-calling, SSE streaming, and full OpenAI API compatibility for Qwen3.5 models on Apple Silicon — no cloud required.**

Built and battle-tested on Mac mini M4 (64GB) and MacBook (16GB).

---

## What This Is

A production-grade local LLM stack that turns any Apple Silicon Mac into an OpenAI-compatible API endpoint with real tool calling:

- **Full OpenAI `/v1/chat/completions` compatibility** — works with any OpenAI client
- **Tool calling** (function calling) via Qwen-Agent XML prompt injection + structured response parsing
- **SSE streaming** with heartbeat keepalives — prevents Telegram/websocket timeouts during long prefills
- **Thinking token suppression** — strips `<think>…</think>` and `Thinking Process:` automatically
- **Metal OOM protection** — proxy token guard warns at 60K tokens, hard-stops at 70K with a retryable user message before the hardware crash zone (~85K on 64GB)
- **Multi-agent support** — remote agents on your LAN can use this machine as their model backend
- **Auto-start on boot** via launchd
- **Full test suite** — 9 tests including tool call selection, argument validation, alias rewriting

Built on [vllm-mlx (waybarrios fork)](https://github.com/waybarrios/vllm-mlx) + a custom async proxy (`aiohttp`).

---

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13+
- Xcode Command Line Tools: `xcode-select --install`
- [Homebrew](https://brew.sh): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- Python 3.11+ via Homebrew: `brew install python`
- `git-lfs` for model downloads: `brew install git-lfs && git lfs install`
- Python packages: `aiohttp`, `requests` (installed in steps below)
- A Qwen3.5 MLX 4-bit model from [mlx-community](https://huggingface.co/mlx-community)

> **Note on `--break-system-packages`:** Commands below use this flag for Homebrew Python 3.11+. If you're using pyenv, conda, or a virtual environment, omit the flag.

> **Note on security:** The proxy listens on `0.0.0.0:8080` with no authentication. If your Mac is on a shared or public network, restrict access with a firewall rule (`sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/bin/python3`), or bind to `127.0.0.1` instead if you don't need LAN access.

---

## Quick Start

### 0. Prerequisites (one-time)

```bash
# Xcode Command Line Tools (required for Homebrew)
xcode-select --install

# Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3.11+ and git-lfs
brew install python git-lfs
git lfs install
```

### 1. Install vllm-mlx

> ⚠️ **Do not use PyPI** (`pip install vllm-mlx`). The PyPI version lacks working tool call parsing. Use the waybarrios fork:

```bash
pip3 install git+https://github.com/waybarrios/vllm-mlx.git --break-system-packages
```

This installs from source and may take a few minutes.

### 2. Download a model

> ⚠️ Models are stored with Git LFS. Without `git lfs install`, `git clone` downloads small pointer files instead of actual weights — the server will fail to load silently.

```bash
mkdir -p ~/mlx-models && cd ~/mlx-models

# Recommended: Qwen3.5-27B (coding + tool use, ~14GB — allow 30-60 min on a typical connection)
git clone https://huggingface.co/mlx-community/Qwen3.5-27B-4bit

# Lighter option: Qwen3.5-9B-Instruct (fast, tool calling, ~5GB, works on 16GB RAM)
git clone https://huggingface.co/mlx-community/Qwen3.5-9B-Instruct-4bit
```

See [Tested Models](#tested-models) below for models confirmed working with this stack, including a Claude-distilled variant with significantly improved reasoning.

### 3. Detect the right startup flag for your model

`Qwen3_5ForConditionalGeneration` appears in ALL Qwen3.5 configs — it does **not** mean the model has a vision tower. Always check the actual weights:

```bash
python3 scripts/detect_flag.py ~/mlx-models/YOUR-MODEL-NAME
```

Output tells you exactly which flag to use:
```
Model:  Qwen3.5-9B-Instruct-4bit
File:   model.safetensors
Weights: 0 vision-related

✅ Use flag:  --continuous-batching
   Engine:    Batched engine (text-only, faster, prefix cache works)

vllm-mlx serve command:
  vllm-mlx serve /path/to/model \
    --host 0.0.0.0 --port 8091 \
    --continuous-batching \
    --tool-call-parser hermes \
    --enable-auto-tool-choice
```

| Model | Vision weights | Flag |
|-------|---------------|------|
| Qwen3.5-27B-4bit | ✅ Yes | `--mllm` |
| Qwen3.5-35B-A3B-4bit | ✅ Yes | `--mllm` |
| Qwen3.5-9B-Instruct-4bit | ❌ No | `--continuous-batching` |
| Qwen3.5-9B-4bit | ❌ No | `--continuous-batching` |

### 4. Configure and start the vllm-mlx server

```bash
mkdir -p ~/bin
cp scripts/start_vllm.sh ~/bin/start_vllm.sh
chmod +x ~/bin/start_vllm.sh
```

Edit `~/bin/start_vllm.sh` — set `MODEL_PATH` and `VLLM_FLAG` to match your model:

```bash
MODEL_PATH="$HOME/mlx-models/Qwen3.5-9B-Instruct-4bit"   # ← your model path
VLLM_FLAG="--continuous-batching"                          # ← from detect_flag.py
```

Then start it:
```bash
~/bin/start_vllm.sh
```

You should see vllm-mlx load the model and start listening on port 8091.

### 5. Configure and start the proxy

```bash
pip3 install aiohttp requests --break-system-packages

mkdir -p ~/mlx-server
cp scripts/proxy.py ~/mlx-server/proxy.py
```

Edit `~/mlx-server/proxy.py` — update the config block at the top:

```python
BACKEND = "http://127.0.0.1:8091"                 # leave as-is
PORT = 8080                                        # leave as-is
LOG_FILE = os.path.expanduser("~/mlx-server/proxy.log")  # leave as-is

MODEL_ALIASES = {
    "qwen9b": "/Users/yourname/mlx-models/Qwen3.5-9B-Instruct-4bit",  # ← required: update this
}
```

> ⚠️ **The `MODEL_ALIASES` entry is required.** If it's missing or the path is wrong, the proxy will fail to route requests. Use the full path, not `~`.

Find your Python path for the launchd step later:
```bash
which python3
# e.g. /opt/homebrew/bin/python3
```

Start the proxy:
```bash
python3 ~/mlx-server/proxy.py
```

### 6. Verify

```bash
# Health check
curl http://localhost:8080/health
# → {"status":"healthy",...}

# Run the full test suite
python3 tests/test_proxy.py --model YOUR-ALIAS
# → 9/9 passed
```

### 7. Manual tool call example

Confirm end-to-end tool calling works with a raw curl:

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR-ALIAS",
    "messages": [{"role": "user", "content": "What files are in /tmp?"}],
    "max_tokens": 200,
    "tools": [{
      "type": "function",
      "function": {
        "name": "list_directory",
        "description": "List files in a directory.",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"]
        }
      }
    }]
  }' | python3 -m json.tool
```

Expected response (trimmed):
```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "tool_calls": [{
        "function": {
          "name": "list_directory",
          "arguments": "{\"path\": \"/tmp\"}"
        }
      }]
    }
  }]
}
```

If you see `"finish_reason": "tool_calls"` and a valid `arguments` JSON string — it's working.

---

## Auto-Start with launchd

Copy the plist templates, then substitute your actual username and Python path:

```bash
cp launchd/com.user.vllm-server.plist ~/Library/LaunchAgents/
cp launchd/com.user.mlx-proxy.plist ~/Library/LaunchAgents/

# Substitute your actual username
sed -i '' "s|/Users/yourname|$HOME|g" ~/Library/LaunchAgents/com.user.vllm-server.plist
sed -i '' "s|/Users/yourname|$HOME|g" ~/Library/LaunchAgents/com.user.mlx-proxy.plist

# Substitute your actual Python path (from 'which python3')
PYTHON_PATH=$(which python3)
sed -i '' "s|/opt/homebrew/bin/python3|$PYTHON_PATH|g" ~/Library/LaunchAgents/com.user.mlx-proxy.plist
```

Load both services:
```bash
launchctl load ~/Library/LaunchAgents/com.user.vllm-server.plist
launchctl load ~/Library/LaunchAgents/com.user.mlx-proxy.plist
```

Restart services:
```bash
launchctl kickstart -k gui/$(id -u)/com.user.vllm-server
launchctl kickstart -k gui/$(id -u)/com.user.mlx-proxy
```

Check logs:
```bash
tail -f ~/mlx-server/vllm-server.log
tail -f ~/mlx-server/proxy.log
```

---

## How Tool Calling Works

Standard `mlx_lm` does not support tool calling. This stack implements it entirely in the proxy — no model modifications required:

1. Request arrives with OpenAI `tools[]` array
2. Proxy converts tools → Qwen-Agent XML system prompt, injects into messages
3. Proxy strips `tools`/`tool_choice`, forces `stream:false` to backend
4. Proxy opens SSE response to client immediately; sends empty heartbeats every 5s
5. Model generates text containing `<tool_call>…</tool_call>` tags
6. Proxy parses tags → OpenAI `tool_calls` format with `finish_reason: "tool_calls"`
7. Client receives standard OpenAI-compatible SSE

The client never sees any of this — it looks identical to the OpenAI API.

---

## Multi-Machine Setup

The proxy binds to `0.0.0.0:8080` by default. Any machine on your LAN can use it:

```bash
# From another machine
curl http://192.168.1.x:8080/health

# Configure your OpenAI client
openai.base_url = "http://192.168.1.x:8080/v1"
```

For OpenClaw agents on a second machine, add a provider:
```json
{
  "baseUrl": "http://192.168.1.x:8080/v1",
  "apiKey": "local",
  "api": "openai-completions"
}
```

---

## Test Suite

```bash
# Test local stack
python3 tests/test_proxy.py --model YOUR-ALIAS

# Test remote machine
python3 tests/test_proxy.py --host 192.168.1.x --model YOUR-ALIAS

# Test with a different port
python3 tests/test_proxy.py --port 9090 --model YOUR-ALIAS
```

Tests: health check · /v1/models · basic chat · thinking tokens stripped · no spurious tool calls · tool call triggered · tool args valid JSON · correct tool selected (multi-tool) · alias rewrite in response

CI-friendly: exits 0 on full pass, 1 on any failure.

---

## Tested Models

Models confirmed working with this stack on Apple Silicon. All require the [waybarrios vllm-mlx fork](https://github.com/waybarrios/vllm-mlx).

| Model | Download | RAM | Flag | Notes |
|-------|----------|-----|------|-------|
| **Qwen3.5-27B-Claude-4.6-Opus-Distilled** | [mlx-community](https://huggingface.co/mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit) | 64GB | `--continuous-batching` | ⭐ **Recommended for 64GB.** Distilled from Claude 4.6 Opus reasoning traces. Noticeably better instruction following and tool use than the base model. No vision weights — use `--continuous-batching`, not `--mllm`. |
| Qwen3.5-27B-4bit | [mlx-community](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | 64GB | `--mllm` | Base model with vision weights. Solid baseline. |
| Qwen3.5-9B-Instruct-4bit | [mlx-community](https://huggingface.co/mlx-community/Qwen3.5-9B-Instruct-4bit) | 16GB | `--continuous-batching` | Best option for 16GB. Fast, good tool calling. |
| Qwen3.5-9B-4bit | [mlx-community](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | 16GB | `--continuous-batching` | Text-only variant of 9B. |
| Qwen3.5-35B-A3B-4bit | [mlx-community](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) | 64GB | `--mllm` | MoE model. Vision weights present. |

**Downloading the recommended distilled model:**
```bash
cd ~/mlx-models
git clone https://huggingface.co/mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit
```
~14GB download. After cloning, run `scripts/detect_flag.py` to confirm the correct flag:
```bash
python3 scripts/detect_flag.py ~/mlx-models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit
# → Use flag: --continuous-batching
```

> **Note on the distilled model:** `Qwen3_5ForConditionalGeneration` appears in this model's config (as with all Qwen3.5 models), but it has **no vision weights** in the actual safetensors files. Always use `detect_flag.py` to check — do not assume from the config alone.

---

## Reference Configurations

### High-performance (Mac mini M4, 64GB) — recommended
- Model: `Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit` · Flag: `--continuous-batching` · Cache: `--cache-memory-percent 0.30` · Prefill: `--chunked-prefill-tokens 1024`

### High-performance (Mac mini M4, 64GB) — base model
- Model: `Qwen3.5-27B-4bit` · Flag: `--mllm` · Cache: `--cache-memory-percent 0.30`

### Standard (MacBook, 16GB)
- Model: `Qwen3.5-9B-Instruct-4bit` · Flag: `--continuous-batching` · Cache: `--cache-memory-percent 0.20`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Model loads but generates garbage | git-lfs not installed before clone — downloaded pointer files | Re-clone after running `git lfs install` |
| `vllm-mlx: command not found` | Install landed in non-PATH location | Run `pip3 show vllm-mlx` to find bin dir; add to PATH |
| `Missing N parameters` on server start | Wrong flag — model has no vision weights but `--mllm` used | Run `detect_flag.py`, use `--continuous-batching` |
| Proxy starts but all requests 502 | vllm-mlx not running or wrong port | Check port 8091: `curl http://localhost:8091/health` |
| Tool calls return empty or plain text | `MODEL_ALIASES` path wrong or missing | Check proxy.py config; use absolute path, not `~` |
| launchd service won't start | Wrong Python path in plist | Run `which python3`, update plist `ProgramArguments` |
| Port 8080 already in use | Another process on that port | Change `PORT` in proxy.py and PORT in plist |
| **Server crashes silently, session lost** | **Metal GPU OOM** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) | **Confirmed crash zone: 51K–85K actual prompt tokens on 64GB.** macOS never fires a memory pressure warning for Metal failures. Set `--chunked-prefill-tokens 1024` (halves peak activation memory). The proxy's token guard (60K warn / 70K hard stop) blocks requests before they reach the crash zone. |
| Proxy token estimate much lower than actual | Char/token ratio varies by content type (2.5–3.1) | Proxy uses `max(chars/3.0, last_known_actual)` — dense tool results (JSON/code) can be 37%+ more tokens than chars/4 estimate. |

---

## Files

| File | Purpose |
|------|---------|
| `scripts/proxy.py` | Async proxy — tool calling, SSE heartbeat, alias rewriting |
| `scripts/start_vllm.sh` | vllm-mlx startup script template |
| `scripts/detect_flag.py` | Detect correct vllm-mlx flag for any model |
| `launchd/com.user.vllm-server.plist` | launchd template for vllm-mlx |
| `launchd/com.user.mlx-proxy.plist` | launchd template for proxy |
| `tests/test_proxy.py` | Full test suite (9 tests) |
| `docs/admin.md` | Detailed admin guide |

---

## Related

- [vllm-mlx (waybarrios fork)](https://github.com/waybarrios/vllm-mlx) — the inference engine
- [mlx-community on HuggingFace](https://huggingface.co/mlx-community) — MLX quantized models
- [OpenClaw](https://openclaw.ai) — the AI agent framework this was built for

---

## License

MIT
