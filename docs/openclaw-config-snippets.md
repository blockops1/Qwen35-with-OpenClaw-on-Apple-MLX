# OpenClaw Configuration Snippets

These are the relevant sections of `openclaw.json` and `.env` needed to wire up the local Qwen3.5 model with LCM. This is not the full config — only the parts that matter for this stack.

---

## 1. Plugin Slot — Activate LCM

In `~/.openclaw/openclaw.json`, add the `plugins` block at the top level:

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "lossless-claw"
    }
  }
}
```

**Why:** Without this, OpenClaw uses its built-in compaction (discards context). With it, LCM takes over — context is summarized and kept retrievable indefinitely.

After adding this, install the plugin and restart the gateway:
```bash
openclaw plugins install @martian-engineering/lossless-claw
openclaw gateway restart
```

---

## 2. Local Model Provider Block

In `~/.openclaw/models.json` (or the `models.providers` section of `openclaw.json`):

```json
{
  "my-local": {
    "baseUrl": "http://localhost:8080/v1",
    "apiKey": "local",
    "api": "openai-completions",
    "models": [
      {
        "id": "qwen35",
        "name": "Qwen3.5 27B (local)",
        "reasoning": false,
        "input": ["text"],
        "cost": {
          "input": 0,
          "output": 0,
          "cacheRead": 0,
          "cacheWrite": 0
        },
        "contextWindow": 85000,
        "maxTokens": 16384
      }
    ]
  }
}
```

**Why `contextWindow: 85000`:** This is the measured safe limit for the 27B model on 64GB RAM — not the model's theoretical context length. LCM uses this number to decide when to compact. Set it too high and LCM compacts too late; set it too low and you waste context capacity.

**Why `maxTokens: 16384`:** The 27B model thinks before responding. Those thinking tokens count against `maxTokens`. If this is set too low, the model can hit the limit mid-think before generating any visible output.

**For a second Mac on LAN:** Add a second provider pointing to the remote machine's IP:
```json
{
  "remote-mac": {
    "baseUrl": "http://<other-mac-ip>:8080/v1",
    "apiKey": "local",
    "api": "openai-completions",
    "models": [ ... same model definition ... ]
  }
}
```

---

## 3. Primary and Fallback Model

In `openclaw.json`, set the primary model and one API fallback for when the local server is unreachable:

```json
{
  "model": {
    "primary": "my-local/qwen35",
    "fallbacks": [
      "google/gemini-2.5-flash"
    ]
  }
}
```

**Why only one fallback:** Each fallback timeout adds ~10 minutes of delay when the local server is unreachable. One fallback is the right balance — you still get a response, but you don't wait 20+ minutes.

**Why local first:** The local model is free and private. API fallback costs money and sends your conversation to a third party. Default to local; fall back only when the server is genuinely down.

---

## 4. LCM Environment Variables

In `~/.openclaw/.env`:

```env
# Use the local model for summarization — free and fast
LCM_SUMMARY_PROVIDER=my-local
LCM_SUMMARY_MODEL=qwen35

# Compact when context reaches 75% of contextWindow
# For contextWindow=85000, this triggers at ~64K tokens
LCM_CONTEXT_THRESHOLD=0.75
```

**Why local for summarization:** If you point LCM at an API model (e.g., Gemini or Claude), every compaction costs money and adds 30–90 seconds of latency. The local Qwen3.5 produces good summaries and is free.

**Why 0.75 threshold:** Gives ~20K token buffer between compaction trigger and the crash zone at ~85K. At 0.85, the buffer shrinks to ~12K tokens — too tight for safety.

---

## Putting It Together — Minimal openclaw.json

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "lossless-claw"
    }
  },
  "model": {
    "primary": "my-local/qwen35",
    "fallbacks": ["google/gemini-2.5-flash"]
  },
  "timeoutSeconds": 600
}
```

With `models.json` (separate file, merged at runtime):
```json
{
  "mode": "merge",
  "providers": {
    "my-local": {
      "baseUrl": "http://localhost:8080/v1",
      "apiKey": "local",
      "api": "openai-completions",
      "models": [
        {
          "id": "qwen35",
          "name": "Qwen3.5 27B (local)",
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
```
