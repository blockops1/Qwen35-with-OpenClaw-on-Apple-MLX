# Local LLM Stack — Admin Guide

**vllm-mlx + Async Proxy for Tool-Calling, SSE Streaming, and OpenAI API Compatibility**

**Last updated:** 2026-03-06  
**Status:** ✅ Operational  
**Tested on:** Apple Silicon (M4 64GB, M-series 16GB) · macOS 15+

---

## What This Is

A production-grade local LLM stack for Apple Silicon that gives you:

- **Full OpenAI API compatibility** — drop-in replacement for `api.openai.com`
- **Tool calling** (function calling) via Qwen-Agent XML prompt injection + response parsing
- **SSE streaming** with heartbeat keepalives (prevents Telegram/websocket timeouts on slow prefills)
- **Thinking token suppression** — strips `<think>…</think>` and `Thinking Process:` from all responses
- **Multi-agent support** — remote agents on LAN can use this machine as their model backend
- **Auto-start on boot** via launchd

Built on [vllm-mlx (waybarrios fork)](https://github.com/waybarrios/vllm-mlx) + a custom async proxy (aiohttp).

---

## Architecture

```
Client (OpenClaw agent, curl, your app)
         │
         │  OpenAI-compatible HTTP  (port 8080)
         ▼
┌─────────────────────────────────────────────────────┐
│  Async Proxy (proxy_qwen35.py — aiohttp)            │
│                                                     │
│  • Model alias rewrite  (friendly name → path)      │
│  • Request size guard (413 if >35k tokens/>300 msg) │
│  • Concurrency limit (429 if >2 requests in flight) │
│  • Tool calling:                                    │
│      - Converts OpenAI tools[] → Qwen-Agent XML     │
│      - Parses <tool_call> tags → OpenAI format      │
│      - Forces stream:false to backend               │
│  • SSE heartbeat every 5s (keeps sockets alive)     │
│  • Strips <think>…</think> from all responses       │
│  • Structured request/response logging              │
└─────────────────────────────────────────────────────┘
         │
         │  localhost (port 8091)
         ▼
┌─────────────────────────────────────────────────────┐
│  vllm-mlx (waybarrios fork)                         │
│                                                     │
│  --mllm                  (VLM models with vision)   │
│  --continuous-batching   (text-only models)         │
│  --tool-call-parser hermes                          │
│  --enable-auto-tool-choice                          │
│  --kv-cache-quantization --kv-cache-quantization-bits 8  │
│  --chunked-prefill-tokens 2048                      │
└─────────────────────────────────────────────────────┘
         │
         ▼
  MLX model weights (~/mlx-models/)
  Qwen3.5 series (4-bit quantized)
```

---

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13+
- Homebrew
- Python 3.11+ (Homebrew recommended)
- A Qwen3.5 MLX 4-bit model from [mlx-community on HuggingFace](https://huggingface.co/mlx-community)

---

## Installation

### 1. Install vllm-mlx (waybarrios fork)

> **Important:** Do NOT install from PyPI (`pip install vllm-mlx`). The PyPI version does not have working tool call parsing. Use the fork:

```bash
pip3 install git+https://github.com/waybarrios/vllm-mlx.git --break-system-packages
```

Verify:
```bash
vllm-mlx serve --help   # should show serve subcommand
```

### 2. Download a model

```bash
# Example: Qwen3.5-27B (good for coding/tool use, ~14GB)
mkdir -p ~/mlx-models
cd ~/mlx-models
git clone https://huggingface.co/mlx-community/Qwen3.5-27B-4bit

# Or smaller/faster option:
git clone https://huggingface.co/mlx-community/Qwen3.5-9B-Instruct-4bit
```

### 3. Determine the correct server flag for your model

`Qwen3_5ForConditionalGeneration` appears in **all** Qwen3.5 model configs regardless of whether they include a vision tower. **Always check the actual weights**, not the config:

```bash
python3 -c "
import json, struct
MODEL = '/path/to/your/mlx-models/MODEL_NAME'
with open(f'{MODEL}/model.safetensors', 'rb') as f:
    hlen = struct.unpack('<Q', f.read(8))[0]
    keys = list(json.loads(f.read(hlen)).keys())
vision = [k for k in keys if 'vision' in k.lower()]
print(f'Vision weights: {len(vision)}')
print(f'Flag to use: {\"--mllm\" if vision else \"--continuous-batching\"}')
"
```

| Vision weights | Flag | Engine type | Notes |
|----------------|------|-------------|-------|
| > 0 | `--mllm` | SimpleEngine | Full VLM (vision + text) |
| 0 | `--continuous-batching` | Batched engine | Text-only; faster, prefix cache works |

**Known models:**
| Model | Flag |
|-------|------|
| Qwen3.5-27B-4bit | `--mllm` |
| Qwen3.5-35B-A3B-4bit | `--mllm` |
| Qwen3.5-9B-Instruct-4bit | `--continuous-batching` |
| Qwen3.5-9B-4bit | `--continuous-batching` |

### 4. Create the startup script

```bash
mkdir -p ~/bin
cat > ~/bin/start-vllm.sh << 'EOF'
#!/bin/bash
# Adjust MODEL_PATH and PORT as needed
# Use --mllm OR --continuous-batching (NOT both) — see guide above

MODEL_PATH="$HOME/mlx-models/Qwen3.5-27B-4bit"  # change this

exec vllm-mlx serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8091 \
  --mllm \                          # or --continuous-batching
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --cache-memory-percent 0.30 \     # adjust for your RAM (see table below)
  --kv-cache-quantization \
  --kv-cache-quantization-bits 8 \
  --chunked-prefill-tokens 2048
EOF
chmod +x ~/bin/start-vllm.sh
```

**Cache memory guidance:**

| Total RAM | Model size | Recommended `--cache-memory-percent` |
|-----------|-----------|--------------------------------------|
| 64GB | 20GB (27B) | 0.30 (19GB cache) |
| 64GB | 14GB (27B 4bit) | 0.35 (22GB cache) |
| 16GB | 5GB (9B) | 0.20 (3.2GB cache) |

### 5. Install and configure the proxy

Download `proxy_qwen35.py` and place it in `~/mlx-server/`:

```bash
mkdir -p ~/mlx-server
# Copy proxy_qwen35.py to ~/mlx-server/proxy_qwen35.py
```

Edit the top of the file to set your model alias and backend port:

```python
BACKEND = "http://127.0.0.1:8091"   # vllm-mlx port
PORT = 8080                          # proxy listen port

MODEL_ALIASES = {
    "qwen35": "/Users/yourname/mlx-models/Qwen3.5-27B-4bit",
    # add more aliases as needed
}
```

Install the proxy dependency:
```bash
pip3 install aiohttp --break-system-packages
```

### 6. Set up auto-start with launchd

**vllm-mlx plist** (`~/Library/LaunchAgents/com.user.vllm-server.plist`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.vllm-server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/Users/yourname/bin/start-vllm.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/yourname/mlx-server/vllm-server.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/yourname/mlx-server/vllm-server.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

**Proxy plist** (`~/Library/LaunchAgents/com.user.mlx-proxy.plist`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.mlx-proxy</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/python3</string>
        <string>/Users/yourname/mlx-server/proxy_qwen35.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/yourname/mlx-server/proxy.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/yourname/mlx-server/proxy.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

Load both:
```bash
launchctl load ~/Library/LaunchAgents/com.user.vllm-server.plist
launchctl load ~/Library/LaunchAgents/com.user.mlx-proxy.plist
```

---

## Verification

```bash
# Health check
curl http://localhost:8080/health
# → {"status":"ok"}

# Models list
curl http://localhost:8080/v1/models

# Basic chat
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen35","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'

# Tool calling
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35",
    "messages": [{"role":"user","content":"What files are in /tmp?"}],
    "max_tokens": 200,
    "tools": [{
      "type": "function",
      "function": {
        "name": "list_dir",
        "description": "List files in a directory",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"]
        }
      }
    }]
  }'
# → finish_reason: "tool_calls", tool_calls: [{function: {name: "list_dir", arguments: '{"path":"/tmp"}'}}]
```

---

## How Tool Calling Works

Standard `mlx_lm` does not support tool calling. This stack implements it entirely in the proxy:

1. **Request arrives** with an OpenAI `tools[]` array
2. **Proxy converts** tools to Qwen-Agent XML system prompt:
   ```
   # Tools
   You may call one or more functions...
   <tools>
   {"name": "...", "description": "...", "parameters": {...}}
   </tools>
   For each function call, return:
   <tool_call>
   {"name": <function-name>, "arguments": <args-json>}
   </tool_call>
   ```
3. **Proxy injects** this as a system message prefix, strips `tools`/`tool_choice` from request, forces `stream:false` to backend
4. **Proxy opens SSE** response to client immediately; sends empty heartbeat chunks every 5s while model prefills
5. **Model generates** raw text containing `<tool_call>…</tool_call>` tags
6. **Proxy parses** tags → OpenAI `tool_calls` array with `finish_reason: "tool_calls"`
7. **Client receives** standard OpenAI-format SSE stream

---

## Request Size Guard

The proxy rejects oversized requests immediately (HTTP 413) rather than passing them to vllm-mlx:

```python
MAX_INPUT_TOKENS = 35000   # estimated from message content (~4 chars/token)
MAX_MESSAGES    = 300      # runaway session guard
```

**Why this matters:** vllm-mlx processes any request regardless of size. A single oversized request (e.g. a runaway chat session with 500+ messages) can take 300+ seconds, hit the server timeout, and hang the model process — blocking all other concurrent sessions until a restart.

The guard returns HTTP 413 in milliseconds. Tune `MAX_INPUT_TOKENS` and `MAX_MESSAGES` for your use case.

## Concurrency Limit

The proxy also enforces a maximum number of simultaneous in-flight backend requests:

```python
MAX_CONCURRENT = 2   # max simultaneous requests to vllm-mlx
```

Requests beyond this limit receive HTTP 429 immediately. This prevents request pile-up from exhausting the KV cache even when individual requests are within the token/message limits.

**Why this matters:** The batched engine allocates KV cache for all concurrent requests simultaneously. Five moderate requests (each 20K tokens) can exhaust 64GB of RAM just as effectively as one 100K-token request. The concurrency limit is the correct protection layer for this.

Log entries: `INFLIGHT_INC`, `INFLIGHT_DEC`, `CONCURRENCY_REJECTED`.

---

**If using OpenClaw**, prevent sessions from reaching the guard in the first place by setting:
```json
"channels": {
  "telegram": { "historyLimit": 20 }
},
"agents": { "defaults": { "compaction": {
  "maxHistoryShare": 0.4
}}}
```
> ⚠️ `historyLimit: 0` means **unlimited** (not zero) in OpenClaw — a common misconfiguration.

---

## Why `stream:false` to Backend

The proxy forces `stream:false` when tools are present because `<tool_call>` tags get split across SSE chunks and can't be reliably parsed mid-stream. The client still gets SSE (with heartbeat keepalives) — only the backend-to-proxy leg is non-streaming.

---

## Log Format

Every request logs structured lines to `proxy.log`:

```
[timestamp] INBOUND method=POST path=/v1/chat/completions body_len=N model=qwen35 is_chat=True
[timestamp] TOOLS_INJECTED count=N stream=sse-heartbeat model_req=/path/to/model   # tool requests
[timestamp] REQUEST messages=N roles=[system:1, user:N, assistant:N] chars=N est_tokens~N tools=True
[timestamp] RESPONSE usage={'prompt_tokens': N, 'completion_tokens': N, 'total_tokens': N}
[timestamp] TOOL_CALLS parsed=['tool_name', ...]
```

Key errors to watch:
| Log entry | Meaning |
|-----------|---------|
| `CONCURRENCY_REJECTED inflight=N limit=N` | Too many concurrent requests — retry in a few seconds |
| `REQUEST_REJECTED est_tokens=N messages=N` | Request exceeded size guard — session too large, start a new one |
| `TOOL_BACKEND_NON200 status=504` | Prefill timed out — context too large or timeout too low |
| `BACKEND_ERROR Server disconnected` | vllm-mlx crashed or wrong flag used (--mllm vs --continuous-batching) |
| `WRITE_ERROR heartbeat` | Client closed connection (usually harmless) |

---

## Remote Access (Multi-Machine)

Any machine on the LAN can use this stack as a model backend. The proxy binds to `0.0.0.0:8080` by default.

From another machine:
```bash
curl http://<this-machine-ip>:8080/health
```

For OpenClaw agents on another machine, configure a provider pointing to `http://<ip>:8080/v1`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Missing N parameters` on startup | Wrong flag — model has no vision weights but `--mllm` used | Check safetensors, use `--continuous-batching` |
| Port conflict on 8080/8091 | Old process still running | `pkill -f mlx_lm.server` or `pkill -f vllm-mlx` |
| Tool calls return empty body | Client parsing raw SSE as JSON | Parse `data: {...}` lines from SSE response |
| HTTP 413 from proxy | Session exceeded size guard (>35k tokens or >300 messages) | Start a new session; reduce `historyLimit` in your client |
| 504 timeout on tool calls | Context too large for timeout setting | Reduce session history; set `compaction.maxHistoryShare: 0.4` in OpenClaw |
| Sessions grow unbounded (OpenClaw) | `historyLimit: 0` means unlimited | Set `historyLimit: 20` in `channels.telegram` |
| Slow generation with two models | Two models loaded in RAM simultaneously | Only load one model at a time |
| `BACKEND_ERROR Server disconnected` | vllm-mlx rejecting stream:false request | Usually wrong flag — check --mllm vs --continuous-batching |

---

## Security Notes

- No authentication on port 8080 or 8091
- Restrict LAN access via firewall if needed
- Proxy logs contain prompt content — rotate periodically
- Never expose port 8091 (vllm-mlx) directly — always go through the proxy

---

## Reference Configurations

### Jill (Mac mini M4, 64GB)
- Model: Qwen3.5-27B-4bit · Flag: `--mllm` · Cache: 30%
- vllm-mlx: port 8091 · Proxy: port 8080
- Remote access: open to LAN for Jack

### Jack (MacBook, 16GB)
- Model: Qwen3.5-9B-Instruct-4bit · Flag: `--continuous-batching` · Cache: 20%
- vllm-mlx: port 8091 · Proxy: port 8080
- Also configured to use Jill's stack as remote fallback

---

## Known Limitations

### `--max-model-len` is not supported in vllm-mlx

The mainline vLLM flag `--max-model-len` does **not** exist in the vllm-mlx fork. Passing it causes:

```
vllm-mlx: error: unrecognized arguments: --max-model-len 40000
```

Use the proxy's `MAX_INPUT_TOKENS` guard instead — it achieves the same protection at the API layer.
