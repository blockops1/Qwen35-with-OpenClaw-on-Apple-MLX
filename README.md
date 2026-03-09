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

## What's New in v1.6

### Model switching

Run two models side-by-side and switch between them with one command:

```bash
~/bin/switch-model.sh distilled    # Qwen3.5 Claude Distilled — conversational, reasoning
~/bin/switch-model.sh instruct     # Qwen3.5 Base Instruct — code generation, software dev
```

- Stops current vllm-mlx + proxy pair, loads the new pair (~10 seconds)
- Session start banner announces active model over Telegram: `🧠 *Active model: Qwen3.5 Claude Distilled (conversational)*`
- Two dedicated launchd plists per model — launchd can never accidentally start a second process
- `scripts/switch-model.sh`, `scripts/start-vllm-distilled.sh`, `scripts/start-vllm-instruct.sh` added

### Session start model announcement

Every new `/new` session now opens with a visible model identity banner:
> 🧠 *Active model: Qwen3.5 Claude Distilled (conversational)*

Set via `MODEL_NAME` env var in the launchd plist — no proxy code changes needed when switching.

---

## What's New in v1.5

### Proxy improvements (battle-tested in production)

**Context token guard** — prevents Metal GPU OOM crashes that silently kill sessions:
- Tracks actual token counts from `usage.prompt_tokens` response field (not just char estimates)
- Uses `max(chars/3.0, last_known_actual)` — conservative ratio validated against dense tool results
- Soft warning at 60K tokens, hard stop at 70K with a user-visible retryable message
- Confirmed crash zone on 64GB: 51K–85K actual prompt tokens

**Accurate token estimation**:
- v1.4 used `chars/4` — too optimistic for code/JSON-heavy tool results (can be chars/2.5)
- v1.5 uses `max(chars/3.0, last_known_actual)` — empirically validated, never underestimates a growing session

**Concurrency fix for large models**:
- `MAX_CONCURRENT` now defaults to `1` for 27B models on 64GB
- Two simultaneous large requests can trigger Metal OOM at lower token counts
- Recommended values: 64GB+27B→1, 64GB+9B→3–4, 32GB+9B→2–3

**Cold-start warning** — client-visible message after 5 minutes of prefill:
- First session prefill at 50k tokens can take 5–10 minutes on 27B
- Proxy now emits a visible SSE warning instead of silent waiting

**Synthetic `/v1/models/{id}` endpoint** — OpenClaw queries this when switching models; vllm-mlx doesn't have it, proxy now handles it

**Separate log file** — proxy now logs to `~/mlx-server/proxy-qwen35.log` (not `proxy.log`), preventing confusion with old proxy logs

### Recommended vllm-mlx flags update

`--chunked-prefill-tokens 1024` added to 27B recommended config — halves peak Metal activation memory during long prefills, confirmed to reduce crash frequency at high token counts.

### OpenClaw configuration guide (new)

See [OpenClaw Agent Configuration](#openclaw-agent-configuration) below.

---

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13+
- Xcode Command Line Tools: `xcode-select --install`
- [Homebrew](https://brew.sh): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- Python 3.11+ via Homebrew: `brew install python`
- `git-lfs` for model downloads: `brew install git-lfs && git lfs install`
- Python packages: `aiohttp`, `psutil`, `requests` (installed in steps below)
- A Qwen3.5 MLX 4-bit model from [mlx-community](https://huggingface.co/mlx-community)

> **Note on `--break-system-packages`:** Commands below use this flag for Homebrew Python 3.11+. If you're using pyenv, conda, or a virtual environment, omit the flag.

> **Note on security:** The proxy listens on `0.0.0.0:8080` with no authentication. If your Mac is on a shared or public network, restrict access with a firewall rule or bind to `127.0.0.1` instead if you don't need LAN access.

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

# Recommended for 64GB: Claude-distilled 27B (~14GB — better reasoning, tool use)
git clone https://huggingface.co/mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit

# Alternative for 64GB: base 27B with vision
git clone https://huggingface.co/mlx-community/Qwen3.5-27B-4bit

# Best option for 16GB: 9B Instruct (~5GB, fast tool calling)
git clone https://huggingface.co/mlx-community/Qwen3.5-9B-Instruct-4bit
```

See [Tested Models](#tested-models) below for the full list.

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
```

| Model | Vision weights | Flag |
|-------|---------------|------|
| Qwen3.5-27B-Claude-4.6-Opus-Distilled | ❌ No | `--continuous-batching` |
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
MODEL_PATH="$HOME/mlx-models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"  # ← your model path
VLLM_FLAG="--continuous-batching"                                               # ← from detect_flag.py
```

Then start it:
```bash
~/bin/start_vllm.sh
```

### 5. Install proxy dependencies and configure

```bash
pip3 install aiohttp psutil requests --break-system-packages

mkdir -p ~/mlx-server
cp scripts/proxy.py ~/mlx-server/proxy.py
```

Edit `~/mlx-server/proxy.py` — update the `MODEL_ALIASES` block at the top:

```python
MODEL_ALIASES = {
    "qwen35": "/Users/yourname/mlx-models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
    # add more aliases here if running multiple models
}
```

> ⚠️ **Use the full absolute path** (not `~`). The alias name is what your OpenClaw agent will use as the model ID.

Find your Python path for the launchd step:
```bash
which python3   # e.g. /opt/homebrew/bin/python3
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

### 7. Manual tool call test

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

If you see `"finish_reason": "tool_calls"` and a valid `arguments` JSON string — it's working.

---

## Auto-Start with launchd

```bash
cp launchd/com.user.vllm-server.plist ~/Library/LaunchAgents/
cp launchd/com.user.mlx-proxy.plist ~/Library/LaunchAgents/

sed -i '' "s|/Users/yourname|$HOME|g" ~/Library/LaunchAgents/com.user.vllm-server.plist
sed -i '' "s|/Users/yourname|$HOME|g" ~/Library/LaunchAgents/com.user.mlx-proxy.plist

PYTHON_PATH=$(which python3)
sed -i '' "s|/opt/homebrew/bin/python3|$PYTHON_PATH|g" ~/Library/LaunchAgents/com.user.mlx-proxy.plist

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
tail -f ~/mlx-server/mlx-vlm-server.log   # vllm-mlx output
tail -f ~/mlx-server/proxy-qwen35.log     # proxy log (v1.5+)
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

## Metal OOM Protection

Apple Silicon's unified memory architecture means the GPU and model share the same RAM pool. When a request exceeds available Metal memory, the process crashes silently — no warning, no log entry, session lost.

**Confirmed crash data (64GB Mac mini M4, 27B model):**
- Last successful request before crash: ~51K actual prompt tokens
- Confirmed crash at: ~85K actual prompt tokens
- Safe operating zone: **≤ 50K prompt tokens**

**How the proxy protects you:**

| Threshold | Action |
|-----------|--------|
| 60K tokens (soft) | Injects visible warning: "Context approaching limit, consider /compact" |
| 70K tokens (hard) | Returns user-visible error: "Context limit reached (~Xk tokens). Run /compact and retry." — no backend call made |

**Key insight:** `--chunked-prefill-tokens 1024` halves peak Metal activation memory during prefill. This is now included in the recommended 27B config and noticeably reduces crash frequency at high token counts.

---

## OpenClaw Agent Configuration

This section covers recommended `openclaw.json` settings for running a local Qwen3.5 agent.

### Provider configuration

Add your local proxy as a custom provider. The `contextWindow` and `maxTokens` are set **on the model definition** — this is where OpenClaw reads them:

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "my-local": {
        "baseUrl": "http://localhost:8080/v1",
        "apiKey": "local",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen35",
            "name": "Qwen 3.5 27B (local)",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 85000,
            "maxTokens": 16384
          }
        ]
      }
    }
  }
}
```

For a **remote machine on your LAN** (e.g., a second agent using this Mac as its model backend), change `baseUrl` to the host machine's LAN IP:

```json
"baseUrl": "http://192.168.1.x:8080/v1"
```

Everything else stays the same — the remote agent doesn't need to know anything about vllm-mlx internals.

### Compaction and timeout settings

```json
{
  "agents": {
    "defaults": {
      "timeoutSeconds": 600,
      "compaction": {
        "mode": "safeguard",
        "reserveTokensFloor": 20000,
        "memoryFlush": {
          "enabled": true,
          "softThresholdTokens": 4000
        }
      }
    }
  }
}
```

**Key values explained:**

| Setting | Value | Why |
|---------|-------|-----|
| `contextWindow` | `85000` | Set on the **model definition** (see above). Matches confirmed crash boundary on 64GB. |
| `maxTokens` | `16384` | Maximum completion tokens per response. |
| `reserveTokensFloor` | `20000` | Hard floor — forces compaction when fewer than 20K tokens remain |
| `softThresholdTokens` | `4000` | Compaction memory-flush runs once context exceeds 4K tokens — intentionally aggressive for local models (no cost, keeps sessions lean) |
| `timeoutSeconds` | `600` | 27B cold-start prefill can take 5–10 min on first request; shorter timeout causes false failures |

### For a 16GB machine (9B model)

```json
{
  "models": {
    "providers": {
      "my-local": {
        "baseUrl": "http://localhost:8080/v1",
        "apiKey": "local",
        "api": "openai-completions",
        "models": [{
          "id": "qwen9b",
          "name": "Qwen 3.5 9B (local)",
          "reasoning": false,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 32768,
          "maxTokens": 8192
        }]
      }
    }
  },
  "agents": {
    "defaults": {
      "timeoutSeconds": 120,
      "compaction": {
        "mode": "safeguard",
        "reserveTokensFloor": 8000,
        "memoryFlush": { "enabled": true, "softThresholdTokens": 4000 }
      }
    }
  }
}
```

### Session settings

```json
{
  "agents": {
    "settings": {
      "maxConcurrent": 2,
      "heartbeat": { "every": "15m" }
    },
    "list": [
      {
        "id": "main",
        "default": true,
        "model": "my-local/qwen35"
      }
    ]
  }
}
```

---

## Model Switching

Two Qwen3.5 27B models are available — only one runs at a time (memory constraint on 64GB):

| Model | Use for | Flag |
|-------|---------|------|
| Qwen3.5 Claude Distilled | Conversational, reasoning, general agent work | `--continuous-batching` |
| Qwen3.5 Base Instruct | Code generation, software development | `--mllm` |

### Setup

Copy the two start scripts and configure your model paths:

```bash
cp scripts/start-vllm-distilled.sh ~/bin/
cp scripts/start-vllm-instruct.sh ~/bin/
chmod +x ~/bin/start-vllm-*.sh

# Edit both scripts — update MODEL_PATH to your actual model directories
```

Copy the launchd plists (one pair per model):

```bash
cp launchd/com.user.vllm-distilled.plist ~/Library/LaunchAgents/
cp launchd/com.user.vllm-instruct.plist ~/Library/LaunchAgents/
cp launchd/com.user.proxy-distilled.plist ~/Library/LaunchAgents/
cp launchd/com.user.proxy-instruct.plist ~/Library/LaunchAgents/

sed -i '' "s|/Users/yourname|$HOME|g" ~/Library/LaunchAgents/com.user.vllm-{distilled,instruct}.plist
sed -i '' "s|/Users/yourname|$HOME|g" ~/Library/LaunchAgents/com.user.proxy-{distilled,instruct}.plist
```

Copy the switch script:

```bash
cp scripts/switch-model.sh ~/bin/
chmod +x ~/bin/switch-model.sh
```

### Switching models

```bash
~/bin/switch-model.sh distilled
~/bin/switch-model.sh instruct
```

The switch takes ~10 seconds. vllm-mlx then loads the new model (2–5 min). Start a `/new` session when ready — the banner confirms which model is active.

---

## Multi-Machine Setup

The proxy binds to `0.0.0.0:8080` by default. Any machine on your LAN can use it:

```bash
# From another machine
curl http://192.168.1.x:8080/health
```

For OpenClaw agents on a second machine, add a provider pointing to the host machine's IP. The proxy handles all routing — the remote agent doesn't need to know anything about vllm-mlx internals.

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
| **Qwen3.5-27B-Claude-4.6-Opus-Distilled** | [mlx-community](https://huggingface.co/mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit) | 64GB | `--continuous-batching` | ⭐ **Recommended for 64GB.** Distilled from Claude 4.6 Opus reasoning traces. Better instruction following and tool use than base model. No vision weights — `--continuous-batching`, not `--mllm`. |
| Qwen3.5-27B-4bit | [mlx-community](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | 64GB | `--mllm` | Base model with vision weights. Solid baseline. |
| Qwen3.5-9B-Instruct-4bit | [mlx-community](https://huggingface.co/mlx-community/Qwen3.5-9B-Instruct-4bit) | 16GB | `--continuous-batching` | Best option for 16GB. Fast, good tool calling. |
| Qwen3.5-9B-4bit | [mlx-community](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) | 16GB | `--continuous-batching` | Text-only variant of 9B. |
| Qwen3.5-35B-A3B-4bit | [mlx-community](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) | 64GB | `--mllm` | MoE model. Vision weights present. |

---

## Reference Configurations

### Mac mini M4, 64GB — Claude-distilled 27B (recommended)

```bash
vllm-mlx serve ~/mlx-models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit \
  --host 0.0.0.0 --port 8091 \
  --continuous-batching \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --cache-memory-percent 0.30 \
  --kv-cache-quantization \
  --kv-cache-quantization-bits 8 \
  --chunked-prefill-tokens 1024 \
  --timeout 600
```

### Mac mini M4, 64GB — base 27B with vision

```bash
vllm-mlx serve ~/mlx-models/Qwen3.5-27B-4bit \
  --host 0.0.0.0 --port 8091 \
  --mllm \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --cache-memory-percent 0.30
```

### MacBook, 16GB — 9B Instruct

```bash
vllm-mlx serve ~/mlx-models/Qwen3.5-9B-Instruct-4bit \
  --host 0.0.0.0 --port 8091 \
  --continuous-batching \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --cache-memory-percent 0.20
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Model loads but generates garbage | git-lfs not installed before clone — pointer files downloaded | Re-clone after `git lfs install` |
| `vllm-mlx: command not found` | Install landed outside PATH | `pip3 show vllm-mlx` → find bin dir → add to PATH |
| `Missing N parameters` on start | Wrong flag — `--mllm` used on text-only model | Run `detect_flag.py`, use `--continuous-batching` |
| Proxy starts but all requests 502 | vllm-mlx not running or wrong port | `curl http://localhost:8091/health` |
| Tool calls return plain text | `MODEL_ALIASES` path wrong or missing | Use absolute path in proxy.py config |
| launchd service won't start | Wrong Python path in plist | `which python3` → update plist `ProgramArguments` |
| Port 8080 already in use | Another process | Change `PORT` in proxy.py and plist |
| Server crashes silently, session lost | Metal GPU OOM (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) | Set `--chunked-prefill-tokens 1024`. Set `contextWindow: 85000` and `softThresholdTokens: 65000` in openclaw.json. The proxy's token guard at 70K will catch most cases, but prevention is better. |
| "Context limit reached" message | Session hit proxy's 70K hard stop | Run `/compact` then retry. Reduce `softThresholdTokens` in openclaw.json to trigger compaction earlier. |
| Cold start takes 5–10 minutes | Normal for first 27B request at 50K+ tokens | Set `timeoutSeconds: 600` in openclaw.json. The proxy will emit a visible warning after 5 min so you know it's still working. |
| Proxy token estimate much lower than actual | Dense tool results (JSON/code) tokenize more densely | v1.5 proxy uses `max(chars/3.0, last_known_actual)` — update from v1.4 if you're still on the old proxy. |

---

## Files

| File | Purpose |
|------|---------|
| `scripts/proxy.py` | Async proxy — tool calling, SSE heartbeat, token guard, model banner |
| `scripts/switch-model.sh` | Switch between distilled and instruct models |
| `scripts/start-vllm-distilled.sh` | vllm-mlx start script — Claude Distilled model |
| `scripts/start-vllm-instruct.sh` | vllm-mlx start script — Base Instruct model |
| `scripts/start_vllm.sh` | Generic vllm-mlx startup template (single-model setup) |
| `scripts/detect_flag.py` | Detect correct vllm-mlx flag for any model |
| `launchd/com.user.vllm-distilled.plist` | launchd plist — distilled vllm-mlx |
| `launchd/com.user.vllm-instruct.plist` | launchd plist — instruct vllm-mlx |
| `launchd/com.user.proxy-distilled.plist` | launchd plist — proxy for distilled model |
| `launchd/com.user.proxy-instruct.plist` | launchd plist — proxy for instruct model |
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
