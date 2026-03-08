# Changelog

## v1.5 (2026-03-08)

### Proxy improvements
- **Context token guard** — soft warning at 60K tokens, hard stop at 70K with retryable user message (prevents silent Metal GPU OOM crash confirmed at ~85K tokens on 64GB)
- **Accurate token estimation** — `max(chars/3.0, last_known_actual)` replaces `chars/4`; validated against dense tool/JSON content
- **Actual token tracking** — updates from `usage.prompt_tokens` each turn for progressively better estimates
- **Concurrency** — `MAX_CONCURRENT` defaults to 1 for 27B on 64GB (2 concurrent requests triggers OOM at high token counts)
- **Cold-start warning** — visible SSE message to client after 5 minutes of prefill
- **Synthetic `/v1/models/{id}` endpoint** — OpenClaw compatibility (vllm-mlx doesn't have this endpoint)
- **Separate log file** — `proxy-qwen35.log` (not `proxy.log`)
- **Backend timeout** — increased to 700s for long 27B prefills

### vllm-mlx reference config
- Added `--kv-cache-quantization` + `--kv-cache-quantization-bits 8` (reduces KV cache memory, recommended for 27B)
- Added `--chunked-prefill-tokens 1024` (halves peak Metal activation memory during prefill)
- Added `--timeout 600`

### OpenClaw configuration guide (new)
- Provider config template with correct key placement (`contextWindow` on model definition)
- Compaction settings verified against two working production machines:
  - `contextWindow: 85000`, `maxTokens: 16384`
  - `reserveTokensFloor: 20000`, `softThresholdTokens: 4000`
  - `timeoutSeconds: 600`
- 16GB / 9B config variant included
- Remote LAN agent setup documented

### Cleanup
- Removed stale duplicate files (`proxy_qwen35.py`, `start-mlx-qwen35.sh`)

---

## v1.4 and earlier

Initial release — basic tool calling, SSE heartbeat, alias rewriting, launchd auto-start, test suite.
