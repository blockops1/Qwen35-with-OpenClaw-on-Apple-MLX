# Setup Guide

This guide covers setting up the vllm-mlx server and proxy from scratch. Read [`PREREQUISITES.md`](PREREQUISITES.md) first.

---

## Directory Layout

After setup, your home directory will look like this:

```
~/
├── mlx-models/
│   └── Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit/   ← model weights
├── mlx-server/
│   ├── proxy.py           ← proxy server (copy from this repo)
│   ├── proxy.log          ← proxy log
│   └── vllm-server.log    ← vllm log
└── bin/
    ├── start-vllm.sh      ← vllm startup script
    ├── start-proxy.sh     ← proxy startup script
    └── switch-model.sh    ← switch between model variants
```

---

## Step 1: Create Directories

```bash
mkdir -p ~/mlx-server ~/bin
```

---

## Step 2: Copy Scripts

```bash
cp scripts/start-vllm.sh ~/bin/start-vllm.sh
cp scripts/start-proxy.sh ~/bin/start-proxy.sh
cp scripts/switch-model.sh ~/bin/switch-model.sh
cp proxy.py ~/mlx-server/proxy.py
chmod +x ~/bin/start-vllm.sh ~/bin/start-proxy.sh ~/bin/switch-model.sh
```

---

## Step 3: Edit the Startup Script

Open `~/bin/start-vllm.sh` and update the model path to match where you downloaded your model:

```bash
# Change this line:
exec vllm-mlx serve "$HOME/mlx-models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit" \

# To your actual model path:
exec vllm-mlx serve "$HOME/mlx-models/YOUR-MODEL-NAME" \
```

Also verify the `--continuous-batching` or `--mllm` flag matches your model. Run `detect-flag.py` if unsure:

```bash
python3 scripts/detect-flag.py ~/mlx-models/YOUR-MODEL-NAME
```

| Flag | When to use |
|------|-------------|
| `--continuous-batching` | Models WITHOUT vision weights (most Instruct + Distilled variants) |
| `--mllm` | Models WITH vision weights (base 27B, 35B) |

---

## Step 4: Set the Model Alias in the Proxy

Open `~/mlx-server/proxy.py` and update `MODEL_BACKENDS` to point to your model:

```python
MODEL_BACKENDS = {
    "qwen35": "/Users/yourname/mlx-models/YOUR-MODEL-NAME",
    # Add more aliases here if running multiple models
}
```

The proxy listens on port 8080 and forwards to vllm-mlx on port 8091. OpenClaw connects to port 8080.

---

## Step 5: Install launchd Auto-Start (optional)

To auto-start vllm-mlx and the proxy on login:

```bash
cp launchd/com.user.vllm.plist ~/Library/LaunchAgents/
cp launchd/com.user.proxy.plist ~/Library/LaunchAgents/
```

Edit both plists — replace every occurrence of `/Users/yourname/` with your actual home directory path.

Then load them:
```bash
launchctl load ~/Library/LaunchAgents/com.user.vllm.plist
launchctl load ~/Library/LaunchAgents/com.user.proxy.plist
```

---

## Step 6: Start Manually (first run)

Start vllm-mlx:
```bash
nohup ~/bin/start-vllm.sh >> ~/mlx-server/vllm-server.log 2>&1 &
```

Wait 2–5 minutes for the model to load. Watch the log:
```bash
tail -f ~/mlx-server/vllm-server.log
# Look for: "Application startup complete."
```

Start the proxy:
```bash
nohup python3 ~/mlx-server/proxy.py >> ~/mlx-server/proxy.log 2>&1 &
```

---

## Step 7: Verify

```bash
# Health check
curl http://localhost:8080/health

# List models
curl http://localhost:8080/v1/models

# Test chat
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen35", "messages": [{"role": "user", "content": "Say hello"}]}'
```

Run the full test suite:
```bash
python3 tests/test_proxy.py
# Expected: 9/9 tests passing
```

---

## Switching Between Models

If you have multiple model variants downloaded:

```bash
# Switch to Claude Distilled (conversational)
~/bin/switch-model.sh distilled

# Switch to Base Instruct (code/software dev)
~/bin/switch-model.sh instruct
```

The switch script unloads the current vllm-mlx + proxy pair and loads the new one. Model load takes 2–5 minutes.

---

## Memory Management

The 27B model uses ~20 GB of Metal memory for weights. KV cache on top of that scales with context length.

**Crash risk:** At ~85K actual tokens of context, Metal OOM can occur silently on 64GB machines. The proxy includes guards:
- Soft warning at configurable threshold
- Hard stop with a user-visible retryable error before the crash zone

The proxy tracks actual token counts reported by vllm-mlx — not character estimates — and emits them as SSE usage chunks so your context manager (OpenClaw LCM or otherwise) sees real numbers. See [`docs/LCM.md`](LCM.md) for how this integrates with Lossless Context Management.

---

## Logs

| Log | Location | What it shows |
|-----|----------|---------------|
| vllm-mlx server | `~/mlx-server/vllm-server.log` | Model load, requests, errors |
| Proxy | `~/mlx-server/proxy.log` | Token counts, tool calls, warnings |

```bash
# Watch both
tail -f ~/mlx-server/vllm-server.log ~/mlx-server/proxy.log
```
