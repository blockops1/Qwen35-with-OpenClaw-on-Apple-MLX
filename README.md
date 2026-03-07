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
- **Multi-agent support** — remote agents on your LAN can use this machine as their model backend
- **Auto-start on boot** via launchd
- **Full test suite** — 9 tests including tool call selection, argument validation, alias rewriting

Built on [vllm-mlx (waybarrios fork)](https://github.com/waybarrios/vllm-mlx) + a custom async proxy (`aiohttp`).

---

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13+
- Homebrew + Python 3.11+ (Homebrew)
- A Qwen3.5 MLX 4-bit model from [mlx-community](https://huggingface.co/mlx-community)

---

## Quick Start

### 1. Install vllm-mlx

> ⚠️ **Do not use PyPI** (`pip install vllm-mlx`). The PyPI version lacks working tool call parsing. Use the waybarrios fork:

```bash
pip3 install git+https://github.com/waybarrios/vllm-mlx.git --break-system-packages
```

### 2. Download a model

```bash
mkdir -p ~/mlx-models && cd ~/mlx-models

# Recommended: Qwen3.5-27B (coding + tool use, ~14GB, needs 32GB+ RAM)
git clone https://huggingface.co/mlx-community/Qwen3.5-27B-4bit

# Lighter option: Qwen3.5-9B-Instruct (fast, tool calling, ~5GB, works on 16GB)
git clone https://huggingface.co/mlx-community/Qwen3.5-9B-Instruct-4bit
```

### 3. Detect the right startup flag for your model

`Qwen3_5ForConditionalGeneration` appears in ALL Qwen3.5 configs — it does **not** mean the model has a vision tower. Always check the actual weights:

```bash
python3 scripts/detect_flag.py ~/mlx-models/YOUR-MODEL-NAME
```

Output tells you exactly which flag to use:
```
Vision weights: 0
✅ Use: --continuous-batching
```

| Model | Vision weights | Flag |
|-------|---------------|------|
| Qwen3.5-27B-4bit | ✅ Yes | `--mllm` |
| Qwen3.5-35B-A3B-4bit | ✅ Yes | `--mllm` |
| Qwen3.5-9B-Instruct-4bit | ❌ No | `--continuous-batching` |
| Qwen3.5-9B-4bit | ❌ No | `--continuous-batching` |

### 4. Configure and start the vllm-mlx server

Edit `scripts/start_vllm.sh` — set your model path and flag:

```bash
cp scripts/start_vllm.sh ~/bin/start_vllm.sh
# Edit MODEL_PATH and VLLM_FLAG
chmod +x ~/bin/start_vllm.sh
~/bin/start_vllm.sh
```

### 5. Configure and start the proxy

```bash
pip3 install aiohttp --break-system-packages

cp scripts/proxy.py ~/mlx-server/proxy.py
# Edit MODEL_ALIASES at the top of the file
python3 ~/mlx-server/proxy.py
```

### 6. Verify

```bash
# Health check
curl http://localhost:8080/health

# Run the full test suite
python3 tests/test_proxy.py --model YOUR-ALIAS
```

---

## Auto-Start with launchd

Copy the plist templates and edit paths:

```bash
cp launchd/com.user.vllm-server.plist ~/Library/LaunchAgents/
cp launchd/com.user.mlx-proxy.plist ~/Library/LaunchAgents/

# Edit both files — replace /Users/yourname with your actual path

launchctl load ~/Library/LaunchAgents/com.user.vllm-server.plist
launchctl load ~/Library/LaunchAgents/com.user.mlx-proxy.plist
```

Restart services:
```bash
launchctl kickstart -k gui/$(id -u)/com.user.vllm-server
launchctl kickstart -k gui/$(id -u)/com.user.mlx-proxy
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
python3 tests/test_proxy.py

# Test remote machine
python3 tests/test_proxy.py --host 192.168.1.x

# Test with specific model alias
python3 tests/test_proxy.py --model qwen9b
```

Tests: health check · /v1/models · basic chat · thinking tokens stripped · no spurious tool calls · tool call triggered · tool args valid JSON · correct tool selected (multi-tool) · alias rewrite in response

CI-friendly: exits 0 on full pass, 1 on any failure.

---

## Reference Configurations

### High-performance (Mac mini M4, 64GB)
- Model: `Qwen3.5-27B-4bit` · Flag: `--mllm` · Cache: `--cache-memory-percent 0.30`

### Standard (MacBook, 16GB)
- Model: `Qwen3.5-9B-Instruct-4bit` · Flag: `--continuous-batching` · Cache: `--cache-memory-percent 0.20`

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
