# Qwen3.5 on Apple Silicon with OpenClaw

Run Qwen3.5 locally on Apple Silicon (M1–M4) as the backend for an [OpenClaw](https://openclaw.ai) agent — with tool calling, context management, and auto-restart.

This repo contains the proxy server, startup scripts, launchd plists, and documentation from a production setup running 24/7 on a 64GB Mac mini M4.

---

## ⚠️ Before You Start: Install git-lfs

HuggingFace model downloads silently download 1KB pointer files if `git-lfs` is not installed. The model directory looks fine. It fails at load time.

```bash
brew install git-lfs && git lfs install
```

Do this before downloading any models. Full prerequisites: [`docs/PREREQUISITES.md`](docs/PREREQUISITES.md)

---

## What's in This Repo

| File / Directory | Purpose |
|-----------------|---------|
| `proxy.py` | OpenAI-compatible proxy — tool call parsing, token tracking, SSE heartbeat, Metal OOM guard |
| `scripts/start-vllm.sh` | Start vllm-mlx with production flags for the 27B model |
| `scripts/start-proxy.sh` | Start the proxy |
| `scripts/switch-model.sh` | Switch between model variants (distilled / instruct) |
| `scripts/detect-flag.py` | Detect the correct vllm-mlx flag (`--mllm` vs `--continuous-batching`) for your model |
| `launchd/` | launchd plists for auto-start on login |
| `tests/test_proxy.py` | 9-test verification suite |
| `requirements.txt` | Python dependencies |
| `docs/PREREQUISITES.md` | Hardware, software, model download — read first |
| `docs/SETUP.md` | Step-by-step setup from scratch |
| `docs/LCM.md` | Lossless Context Management integration |
| `docs/openclaw-config-snippets.md` | OpenClaw config blocks (plugin slot, model provider, LCM env vars) |

---

## Quick Start

1. Read [`docs/PREREQUISITES.md`](docs/PREREQUISITES.md) — hardware requirements, git-lfs, vllm-mlx install
2. Download a model (27B on 64GB, 9B on 32GB)
3. Follow [`docs/SETUP.md`](docs/SETUP.md)
4. Verify with `python3 tests/test_proxy.py` — expect 9/9 passing
5. Point OpenClaw at `http://localhost:8080/v1` — see [`docs/openclaw-config-snippets.md`](docs/openclaw-config-snippets.md)

---

## The Proxy

The proxy sits between OpenClaw and vllm-mlx and handles several things that vllm-mlx doesn't do natively:

**Tool calling:** Qwen3.5 uses Qwen-Agent XML format for tool calls (`<tool_call>` tags). The proxy injects the required system prompt and rewrites XML responses into OpenAI `tool_calls` JSON that OpenClaw can parse.

**Token tracking:** The proxy reports actual `prompt_tokens` back to OpenClaw as trailing SSE usage chunks — including the tokens injected for tool calling. Without this, the context manager sees a ~10K token undercount and compacts too late, which on a local model means Metal OOM rather than a graceful error.

**Metal OOM guard:** At ~85K actual tokens on 64GB, the Metal GPU runs out of memory silently. The proxy tracks actual token counts and sends a user-visible retryable error before the crash zone.

**SSE heartbeat:** Empty SSE chunks every 5 seconds during long prefills prevent Telegram (and other clients) from dropping the socket on responses > 60s.

---

## Models

| Model | Size | Flag | Best for |
|-------|------|------|---------|
| Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit | ~14 GB | `--continuous-batching` | Conversational, reasoning, general use |
| Qwen3.5-27B-4bit | ~16 GB | `--mllm` | Vision tasks, base model |
| Qwen3.5-9B-Instruct-4bit | ~5 GB | `--continuous-batching` | 32GB machines, code generation |

Use `scripts/detect-flag.py` to confirm the right flag for any model — the `Qwen3_5ForConditionalGeneration` architecture name appears in all Qwen3.5 configs regardless of whether vision weights are present. Don't trust the config; check the actual weights.

---

## Context Management (LCM)

For long-running agent sessions, install the [Lossless Claw](https://github.com/martian-engineering/lossless-claw) plugin for OpenClaw. It summarizes old context into a retrievable DAG instead of discarding it — essential for local models where you have a hard memory wall rather than a soft token limit.

See [`docs/LCM.md`](docs/LCM.md) for integration details and tuning.

---

## Tested On

- Mac mini M4 64GB · macOS 26.3.1
- vllm-mlx 0.2.6 (waybarrios fork)
- Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit
- OpenClaw 2026.3.x + lossless-claw 0.2.5

---

## License

MIT
