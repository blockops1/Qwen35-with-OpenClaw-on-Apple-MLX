# Lossless Context Management (LCM)

LCM is an OpenClaw plugin that replaces the built-in context compaction with a lossless approach: instead of discarding old context when a session gets long, it summarizes it into a DAG and keeps everything retrievable. You never lose conversation history.

For local models, LCM matters more than it does for API models — because local models have a hard memory wall (Metal OOM) rather than a soft token limit.

---

## Why This Stack Needs LCM

Without context management, a 27B model on 64GB runs into Metal GPU OOM at around 85K actual tokens of context. This crash is silent — no error message, just a dead server.

The proxy handles the immediate guard (hard stop before the crash zone), but LCM is the longer-term solution: it compacts old context before the session grows long enough to be dangerous, and it uses the local model to do the summarization (free, fast).

---

## Install

```bash
openclaw plugins install @martian-engineering/lossless-claw
```

After install, restart the OpenClaw gateway for the plugin to take effect.

---

## Configuration

### 1. Activate in openclaw.json

Add the plugin slot to your OpenClaw config:

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "lossless-claw"
    }
  }
}
```

### 2. Set Environment Variables

Add to `~/.openclaw/.env`:

```env
# Model to use for LCM summarization — use your local model (free)
LCM_SUMMARY_PROVIDER=my-local
LCM_SUMMARY_MODEL=qwen35

# Compact when context reaches this fraction of the model's contextWindow
# 0.75 = compact at 75% full. Adjust based on your crash threshold.
LCM_CONTEXT_THRESHOLD=0.75
```

See [`openclaw-config-snippets.md`](openclaw-config-snippets.md) for the full provider + model config needed to wire up `my-local`.

### 3. Set contextWindow on the Local Model

In your `models.json` or provider config, set `contextWindow` to match the actual safe limit of your model — **not** the model's theoretical maximum. For the 27B on 64GB:

```json
{
  "id": "qwen35",
  "contextWindow": 85000,
  "maxTokens": 16384
}
```

LCM uses `contextWindow` to calculate when to compact. If you leave it at the default (200K+), LCM will compact far too late and you'll hit Metal OOM first.

---

## Token Tracking — Critical Detail

LCM makes compaction decisions based on the token count it receives from the model. This count must be accurate.

The proxy in this repo emits a trailing SSE `usage` chunk after every response — including tool-call responses — so the token count OpenClaw sees matches what vllm-mlx actually processed. Without this, the count can be ~10K tokens lower than reality (because the proxy injects system prompts for tool calling that OpenClaw doesn't account for in its own estimate).

You can verify this is working by watching the proxy log:

```bash
grep "USAGE_CHUNK_SENT" ~/mlx-server/proxy.log
# Expected: USAGE_CHUNK_SENT prompt=N completion=M  (after each tool call)
```

---

## Summarization Model

LCM needs a model to generate summaries. Point it at your local model so summarization is free:

```env
LCM_SUMMARY_PROVIDER=my-local
LCM_SUMMARY_MODEL=qwen35
```

If the local model is unavailable and you fall back to an API model for summarization, LCM summaries will cost money and be slower. The proxy's SSE heartbeat (every 5 seconds) prevents Telegram timeouts during long summarization runs.

---

## Retrieving Compacted Context

LCM stores summaries in a SQLite database at `~/.openclaw/lcm.db`. These are searchable from within an OpenClaw session using the `lcm_grep`, `lcm_describe`, and `lcm_expand_query` tools — you can always retrieve details from any prior session, even after compaction.

---

## Tuning Reference

| Setting | Conservative (crash-safe) | Balanced | Aggressive |
|---------|--------------------------|----------|------------|
| `contextWindow` | 60,000 | 85,000 | 100,000 |
| `LCM_CONTEXT_THRESHOLD` | 0.65 | 0.75 | 0.85 |
| Effective compact-at | ~39K tokens | ~64K tokens | ~85K tokens |

Start conservative. The 27B model on 64GB crashed in testing at 85,524 actual tokens — that's the measured hard wall, not a theoretical estimate.
