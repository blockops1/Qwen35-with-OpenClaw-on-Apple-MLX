# Changelog

## v2.0 (2026-03-10)

Major update — production-validated stack with LCM integration and proxy token tracking fix.

### Proxy (`proxy.py`)
- **Token tracking fix:** Proxy now emits a trailing SSE `usage` chunk after every tool-call response, including the actual `prompt_tokens` count from vllm-mlx. Without this, the context manager sees a ~10K token undercount (because the proxy injects Qwen-Agent XML system prompts for tool calling that aren't visible to OpenClaw). Symptom was LCM compacting too late → Metal OOM.
- **SSE heartbeat:** Empty SSE chunks every 5 seconds during long prefills — prevents Telegram/client socket timeouts on responses > 60s.
- **Metal OOM guard:** Soft warning and hard stop (with user-visible retryable error) before the crash zone. Measured crash at 85,524 actual tokens on 64GB.
- **Accurate token estimation:** Uses `max(chars/3.0, last_known_actual)` — validated against dense tool/JSON content. Updates from `usage.prompt_tokens` each turn.
- **Session start banner:** Every new session announces the active model once.
- **Synthetic `/v1/models/{id}` endpoint:** OpenClaw compatibility (vllm-mlx doesn't expose per-model detail endpoints).

### Documentation
- `docs/PREREQUISITES.md` — hardware requirements, git-lfs warning, Python/vllm-mlx install, model download, Telegram bot setup, multi-machine networking
- `docs/SETUP.md` — complete setup guide from scratch
- `docs/LCM.md` — Lossless Context Management integration: why it matters for local models, config, token tracking detail, tuning reference
- `docs/openclaw-config-snippets.md` — four annotated config snippets: plugin slot, local model provider, primary/fallback model, LCM env vars
- `requirements.txt` — pinned Python dependencies

### vllm-mlx Reference Config
- `--kv-cache-quantization --kv-cache-quantization-bits 8` — reduces KV cache memory
- `--chunked-prefill-tokens 1024` — halves peak Metal activation memory during prefill
- `--timeout 600` — for long 27B prefills
- `--cache-memory-percent 0.30` — explicit KV cache allocation

### Scripts
- `scripts/switch-model.sh` — switch between model variants (distilled / instruct) with one command; handles launchd unload/load of both vllm and proxy pairs
- `scripts/start-vllm.sh` — updated with full production flags
- `scripts/start-proxy.sh` — proxy startup wrapper

### launchd
- Simplified to one vllm plist + one proxy plist (generic names)
- `RunAtLoad=true` + `KeepAlive=true` — auto-restart on crash

---

## v1.6 (2026-03-09)

- `scripts/switch-model.sh` — switch between distilled and instruct with one command
- `scripts/start-vllm-distilled.sh` / `start-vllm-instruct.sh` — dedicated start scripts per model
- Four launchd plists (two per model)
- Session start model announcement banner

## v1.5 (2026-03-08)

- Context token guard (soft warning + hard stop)
- Accurate token estimation
- Concurrency limit for 27B on 64GB
- Cold-start warning after 5 min prefill
- OpenClaw configuration guide

## v1.4 and earlier

Initial release — basic tool calling, SSE heartbeat, alias rewriting, launchd auto-start, test suite.
