"""
MLX-VLM Async Proxy for Jill
- Maps model alias 'qwen35' to local model path
- Injects Qwen-Agent system prompt for tool calling
- Parses <tool_call> tags and converts to OpenAI tool_calls format
- Streams SSE heartbeat chunks during tool requests to keep Telegram socket alive
Port 8080 → forwards to vllm-mlx on port 8091

Log file: /Users/jill/mlx-server/proxy-qwen35.log
"""
import asyncio
import json
import re
import uuid
import datetime
import aiohttp
import psutil
from aiohttp import web

DEFAULT_BACKEND = "http://127.0.0.1:8091"
PORT = 8080
LOG_FILE = "/Users/jill/mlx-server/proxy-qwen35.log"
HEARTBEAT_INTERVAL = 5   # seconds between SSE keepalive chunks
COLD_START_WARN_S  = 60  # seconds before emitting cold-start warning to client

MODEL_ALIASES = {
    "qwen35":           "/Users/jill/mlx-models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",  # active: distilled 2026-03-07
    "qwen35-base":      "/Users/jill/mlx-models/Qwen3.5-27B-4bit",                                  # inactive: base VLM
}
# Per-alias backend override — only needed when running a second server on a different port
MODEL_BACKENDS = {}

# Request size limits — prevent runaway sessions from hanging the model
MAX_INPUT_TOKENS = 35000   # ~140k chars; hard reject above this
MAX_MESSAGES    = 300      # runaway session guard

# Concurrency limit — prevent request pile-up from saturating vllm-mlx
MAX_CONCURRENT  = 2        # max simultaneous in-flight backend requests
# Tune by hardware: 64GB+27B→2, 64GB+9B→4, 32GB+9B→2-3 (429 → Claude fallback)
_inflight       = 0        # current count (guarded by asyncio single-thread)

# Memory warning thresholds — warnings only, no enforcement (testing mode)
# Tiers are fractions of TOTAL_RAM, measured at startup — portable across machines.
TOTAL_RAM_BYTES = psutil.virtual_memory().total
TOTAL_RAM_GB    = TOTAL_RAM_BYTES / (1024 ** 3)  # GiB — matches OS/Apple reported value

MEM_WARN_TIERS = [
    # (available_fraction_of_total, short_label, full_message)  — highest severity first
    # Calibrated for model-loaded baseline (~77% used on 64GB, ~60% on 128GB)
    (0.05, "Crash likely imminent — save work NOW",
           "🚨 *Memory CRITICAL* — only {avail:.1f} GB free ({pct:.0f}% of {total:.0f} GB). Crash likely imminent. Save work now."),
    (0.10, "High pressure — /compact or /new soon",
           "🔴 *Memory HIGH* — {avail:.1f} GB free ({pct:.0f}% of {total:.0f} GB). High pressure — consider /compact or /new soon."),
    (0.15, "Session growing large",
           "⚠️ *Memory WARNING* — {avail:.1f} GB free ({pct:.0f}% of {total:.0f} GB). Session growing large."),
    (0.20, "Session is getting long",
           "⚡ *Memory NOTICE* — {avail:.1f} GB free ({pct:.0f}% of {total:.0f} GB). Session is getting long."),
]

ALIAS_REVERSE = {v: k for k, v in MODEL_ALIASES.items()}

QWEN_TOOL_SYSTEM = """\
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{tool_list}
</tools>

For each function call, return a json object with function name and arguments \
within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def make_session_start_notice(avail_gb: float) -> str:
    """
    Startup notice injected into the first response of every session.
    Lists warning tiers calculated from TOTAL_RAM_GB so they auto-scale per machine.
    """
    lines = ["⚙️ *[TESTING MODE] Compaction is disabled.*",
             f"This session will grow until memory runs out. "
             f"Start a new session (/new) before you hit the limit.\n",
             f"Memory warnings will appear at (machine total: {TOTAL_RAM_GB:.0f} GB):"]
    # Print tiers least-severe → most-severe
    for frac, label, msg_template in reversed(MEM_WARN_TIERS):
        threshold_gb = TOTAL_RAM_GB * frac
        emoji = msg_template.split()[0]
        lines.append(f"  {emoji} < {frac*100:.0f}% free  (< {threshold_gb:.1f} GB)  — {label}")
    lines.append(f"\nCurrent: {avail_gb:.1f} GB free of {TOTAL_RAM_GB:.0f} GB "
                 f"({avail_gb/TOTAL_RAM_GB*100:.0f}% free)\n"
                 "─────────────────────────────────")
    return "\n".join(lines) + "\n\n"


def get_memory_warning() -> tuple[str | None, float]:
    """
    Check system available RAM as fraction of total (measured at startup).
    Returns (warning_text, available_gb) or (None, available_gb).
    No enforcement — warnings only during testing phase.
    """
    try:
        mem = psutil.virtual_memory()
        available_gb   = mem.available / (1024 ** 3)  # GiB
        available_frac = mem.available / TOTAL_RAM_BYTES
        used_pct       = mem.percent
        log(f"MEM_CHECK available={available_gb:.1f}GB ({available_frac*100:.0f}% free) "
            f"used={used_pct:.0f}% total={TOTAL_RAM_GB:.0f}GB")
        for threshold_frac, _label, msg_template in MEM_WARN_TIERS:
            if available_frac < threshold_frac:
                msg = msg_template.format(avail=available_gb, total=TOTAL_RAM_GB, pct=available_frac*100)
                log(f"MEM_WARN tier=<{threshold_frac*100:.0f}% available={available_gb:.1f}GB")
                return msg, available_gb
        return None, available_gb
    except Exception as e:
        log(f"MEM_CHECK_ERROR {e}")
        return None, 0.0


def estimate_tokens(char_count):
    return char_count // 4


def build_tool_system_prompt(tools: list) -> str:
    tool_defs = []
    for t in tools:
        if t.get("type") == "function":
            tool_defs.append(json.dumps(t["function"], ensure_ascii=False))
    return QWEN_TOOL_SYSTEM.format(tool_list="\n".join(tool_defs))


def inject_tool_prompt(messages: list, tool_system: str) -> list:
    messages = [m.copy() for m in messages]
    if messages and messages[0].get("role") == "system":
        existing = messages[0].get("content", "")
        messages[0]["content"] = tool_system + "\n\n" + existing
    else:
        messages.insert(0, {"role": "system", "content": tool_system})
    return messages


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^Thinking Process:.*?(\n\n|\Z)", "", text, flags=re.DOTALL).strip()
    return text


def parse_tool_calls(content: str):
    """
    Extract <tool_call>...</tool_call> blocks.
    Returns (cleaned_content, tool_calls_list | None).
    """
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    matches = pattern.findall(content)
    if not matches:
        return strip_thinking(content), None

    tool_calls = []
    for raw in matches:
        try:
            parsed = json.loads(raw)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": parsed.get("name", ""),
                    "arguments": json.dumps(parsed.get("arguments", {})),
                },
            })
        except json.JSONDecodeError:
            log(f"TOOL_PARSE_ERROR raw={raw[:100]}")

    cleaned = strip_thinking(pattern.sub("", content).strip())
    return cleaned, tool_calls if tool_calls else None


def rewrite_response(resp_data: dict) -> dict:
    try:
        choice = resp_data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        cleaned_content, tool_calls = parse_tool_calls(content)
        if tool_calls:
            message["content"] = cleaned_content or None
            message["tool_calls"] = tool_calls
            choice["finish_reason"] = "tool_calls"
            log(f"TOOL_CALLS parsed={[tc['function']['name'] for tc in tool_calls]}")
        else:
            message["content"] = cleaned_content
        resp_data["choices"][0]["message"] = message
    except Exception as e:
        log(f"REWRITE_ERROR {e}")
    return resp_data


async def fetch_backend_blocking(body_bytes: bytes, headers: dict, path_qs: str, alias: str = None):
    """Non-streaming backend fetch. Returns (bytes, status_int)."""
    backend = MODEL_BACKENDS.get(alias, DEFAULT_BACKEND)
    url = backend + path_qs
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method="POST",
                url=url,
                headers=headers,
                data=body_bytes,
                timeout=aiohttp.ClientTimeout(total=700),
            ) as resp:
                data = await resp.read()
                return data, resp.status
    except Exception as e:
        log(f"BACKEND_ERROR {e}")
        return json.dumps({"error": str(e)}).encode(), 502


def make_sse_chunk(request_id, model, delta, finish_reason=None):
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(datetime.datetime.now().timestamp()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


async def safe_write(response: web.StreamResponse, data: bytes, label: str = "") -> bool:
    """Write to SSE stream, silently swallow client-disconnect errors. Returns False if write failed."""
    try:
        await response.write(data)
        return True
    except Exception as e:
        log(f"WRITE_ERROR{' ' + label if label else ''} {e}")
        return False


async def handle_tool_stream(request: web.Request, data: dict, headers: dict, body_bytes: bytes, session_start: bool = False):
    """
    Tool-call path:
    1. Open SSE stream to client immediately
    2. Send empty heartbeat chunks every HEARTBEAT_INTERVAL seconds
    3. Backend request runs concurrently (non-streaming)
    4. When done, emit real response as SSE + [DONE]
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    model_alias = data.get("model", "qwen35")

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    # Session start notice — injected into first response only
    mem_warning, avail_gb = get_memory_warning()
    if session_start:
        notice = make_session_start_notice(avail_gb)
        notice_chunk = make_sse_chunk(request_id, model_alias, {"role": "assistant", "content": notice})
        await safe_write(response, notice_chunk, "session-start-notice")
    elif mem_warning:
        # Memory warning — only if not already showing session start notice
        warn_chunk = make_sse_chunk(request_id, model_alias, {"role": "assistant", "content": f"{mem_warning}\n\n"})
        await safe_write(response, warn_chunk, "mem-warn")

    # Launch backend request concurrently
    backend_task = asyncio.create_task(
        fetch_backend_blocking(body_bytes, headers, request.path_qs, alias=model_alias)
    )

    # Send heartbeat chunks until backend responds
    # After COLD_START_WARN_S seconds, emit a visible warning to the client
    elapsed_heartbeat = 0
    cold_start_warned  = False
    try:
        while not backend_task.done():
            heartbeat = make_sse_chunk(request_id, model_alias, {"content": ""})
            if not await safe_write(response, heartbeat, "heartbeat"):
                break  # client gone, but let backend_task finish so we can log it
            try:
                await asyncio.wait_for(asyncio.shield(backend_task), timeout=float(HEARTBEAT_INTERVAL))
                break  # backend done
            except asyncio.TimeoutError:
                elapsed_heartbeat += HEARTBEAT_INTERVAL
                if not cold_start_warned and elapsed_heartbeat >= COLD_START_WARN_S:
                    cold_start_warned = True
                    warning_text = (
                        "⏳ *Cold start detected* — initial prefill in progress. "
                        "This can take up to 5 minutes on the first request. "
                        "Subsequent responses in this session will be much faster."
                    )
                    warn_chunk = make_sse_chunk(request_id, model_alias, {"role": "assistant", "content": warning_text})
                    await safe_write(response, warn_chunk, "cold-start-warn")
                    log(f"COLD_START_WARN sent after {elapsed_heartbeat}s")
                continue  # still waiting, loop again
    except Exception as e:
        log(f"HEARTBEAT_ERROR {e}")

    # Retrieve backend result
    resp_bytes, resp_status = await backend_task

    if resp_status != 200:
        log(f"TOOL_BACKEND_NON200 status={resp_status} body={resp_bytes[:300]!r}")
        err = make_sse_chunk(request_id, model_alias, {"content": f"[backend error {resp_status}]"}, "stop")
        await safe_write(response, err, "error-chunk")
        await safe_write(response, b"data: [DONE]\n\n", "done")
        return response

    try:
        resp_data = json.loads(resp_bytes)
        usage = resp_data.get("usage", {})
        if usage:
            log(f"RESPONSE usage={usage}")

        # Alias rewrite in response
        resp_str = json.dumps(resp_data)
        for path, alias in ALIAS_REVERSE.items():
            resp_str = resp_str.replace(path, alias)
        resp_data = json.loads(resp_str)

        resp_data = rewrite_response(resp_data)
        choice = resp_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")
        tool_calls = message.get("tool_calls")
        content = message.get("content") or ""

        if tool_calls:
            # Emit tool_calls delta
            delta = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            }
            await safe_write(response, make_sse_chunk(request_id, model_alias, delta, "tool_calls"), "tool-delta")
        else:
            # Emit content then stop
            if content:
                await safe_write(response, make_sse_chunk(request_id, model_alias, {"role": "assistant", "content": content}), "content")
            await safe_write(response, make_sse_chunk(request_id, model_alias, {}, "stop"), "stop-chunk")

    except Exception as e:
        log(f"STREAM_EMIT_ERROR {e}")

    await safe_write(response, b"data: [DONE]\n\n", "done")
    return response


async def handle(request: web.Request) -> web.Response:
    body_bytes = await request.read()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "transfer-encoding")}

    data = None
    if body_bytes:
        try:
            data = json.loads(body_bytes)
        except Exception as e:
            log(f"BODY_PARSE_ERROR method={request.method} path={request.path_qs} body_len={len(body_bytes)} err={e}")

    is_chat = data is not None and request.path == "/v1/chat/completions" and request.method == "POST"

    requested_model = data.get("model", "<none>") if data else "<no-body>"
    log(f"INBOUND method={request.method} path={request.path_qs} "
        f"body_len={len(body_bytes)} model={requested_model} is_chat={is_chat} peer={request.remote}")

    # Handle /v1/models/{alias} — OpenClaw hits this when switching models
    # vllm-mlx doesn't have this endpoint; return a synthetic model object
    if request.method == "GET" and request.path.startswith("/v1/models/"):
        model_id = request.path[len("/v1/models/"):]
        import time
        model_obj = {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "jill-local",
        }
        log(f"MODELS_SYNTHETIC id={model_id} → 200")
        return web.Response(
            status=200,
            content_type="application/json",
            body=json.dumps(model_obj).encode()
        )

    if not is_chat:
        # Passthrough for health checks, models list, etc.
        url = DEFAULT_BACKEND + request.path_qs
        log(f"PASSTHROUGH → {url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    data=body_bytes or None,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    resp_bytes = await resp.read()
                    resp_headers = {k: v for k, v in resp.headers.items()
                                    if k.lower() in ("content-type",)}
                    log(f"PASSTHROUGH_RESP status={resp.status} body_len={len(resp_bytes)} "
                        f"body_preview={resp_bytes[:200]!r}")
                    return web.Response(status=resp.status, body=resp_bytes, headers=resp_headers)
        except Exception as e:
            log(f"BACKEND_ERROR {e}")
            return web.Response(status=502, body=json.dumps({"error": str(e)}).encode())

    # --- Chat completions ---
    messages = data.get("messages", [])
    tools = data.pop("tools", None)
    data.pop("tool_choice", None)

    # Concurrency guard — reject if too many requests already in flight
    global _inflight
    if _inflight >= MAX_CONCURRENT:
        log(f"CONCURRENCY_REJECTED inflight={_inflight} limit={MAX_CONCURRENT} messages={len(messages)}")
        return web.Response(
            status=429,
            content_type="application/json",
            body=json.dumps({
                "error": {
                    "message": (
                        f"Model busy: {_inflight} requests already in progress "
                        f"(limit {MAX_CONCURRENT}). Retry in a few seconds."
                    ),
                    "type": "rate_limit_exceeded",
                    "code": 429,
                }
            }).encode()
        )

    # Request size guard — fail fast before model hangs
    total_chars_guard = sum(len(m.get("content", "") or "") for m in messages)
    est_tokens_guard  = estimate_tokens(total_chars_guard)
    if est_tokens_guard > MAX_INPUT_TOKENS or len(messages) > MAX_MESSAGES:
        log(f"REQUEST_REJECTED est_tokens={est_tokens_guard} messages={len(messages)} "
            f"— exceeds limits (max_tokens={MAX_INPUT_TOKENS} max_messages={MAX_MESSAGES})")
        return web.Response(
            status=413,
            content_type="application/json",
            body=json.dumps({
                "error": {
                    "message": (
                        f"Request too large: ~{est_tokens_guard} tokens / {len(messages)} messages. "
                        f"Limits: {MAX_INPUT_TOKENS} tokens, {MAX_MESSAGES} messages. "
                        "Start a new session (/new)."
                    ),
                    "type": "request_too_large",
                    "code": 413,
                }
            }).encode()
        )

    # Session start detection — first user message (system + 1 user msg = new session)
    user_msgs = [m for m in messages if m.get("role") == "user"]
    is_session_start = len(user_msgs) == 1
    if is_session_start:
        log("SESSION_START detected — will inject startup notice")

    if tools:
        tool_system = build_tool_system_prompt(tools)
        messages = inject_tool_prompt(messages, tool_system)
        data["messages"] = messages
        data["stream"] = False  # backend is non-streaming; we fake SSE to client
        data["enable_thinking"] = False
        if data.get("model") in MODEL_ALIASES:
            data["model"] = MODEL_ALIASES[data["model"]]

        total_chars = sum(len(m.get("content", "") or "") for m in messages)
        est_tokens = estimate_tokens(total_chars)
        roles = [m.get("role", "?") for m in messages]
        role_summary = ", ".join(f"{r}:{roles.count(r)}" for r in dict.fromkeys(roles))
        log(f"TOOLS_INJECTED count={len(tools)} stream=sse-heartbeat model_req={data.get('model','?')}")
        log(f"REQUEST messages={len(messages)} roles=[{role_summary}] "
            f"chars={total_chars} est_tokens~{est_tokens} "
            f"max_tokens={data.get('max_tokens','?')} tools=True")

        body_bytes = json.dumps(data).encode()
        _inflight += 1
        log(f"INFLIGHT_INC count={_inflight}")
        try:
            return await handle_tool_stream(request, data, headers, body_bytes, session_start=is_session_start)
        finally:
            _inflight -= 1
            log(f"INFLIGHT_DEC count={_inflight}")

    # No tools — regular passthrough (streaming or not, as client requests)
    original_model = data.get("model", "<none>")
    if data.get("model") in MODEL_ALIASES:
        data["model"] = MODEL_ALIASES[data["model"]]
        log(f"MODEL_ALIAS {original_model} → {data['model']}")
    elif data.get("model") not in MODEL_ALIASES.values():
        log(f"MODEL_UNKNOWN model={original_model} — passing through as-is")
    data["enable_thinking"] = False

    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    est_tokens = estimate_tokens(total_chars)
    roles = [m.get("role", "?") for m in messages]
    role_summary = ", ".join(f"{r}:{roles.count(r)}" for r in dict.fromkeys(roles))
    log(f"REQUEST messages={len(messages)} roles=[{role_summary}] "
        f"chars={total_chars} est_tokens~{est_tokens} "
        f"max_tokens={data.get('max_tokens','?')} tools=False")

    body_bytes = json.dumps(data).encode()
    backend = MODEL_BACKENDS.get(original_model, DEFAULT_BACKEND)
    url = backend + request.path_qs

    _inflight += 1
    log(f"INFLIGHT_INC count={_inflight}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=request.method,
                url=url,
                headers=headers,
                data=body_bytes,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                resp_bytes = await resp.read()
                try:
                    resp_data = json.loads(resp_bytes)
                    usage = resp_data.get("usage", {})
                    if usage:
                        log(f"RESPONSE usage={usage}")
                    # Strip thinking tokens from non-tool responses
                    for choice in resp_data.get("choices", []):
                        msg = choice.get("message", {})
                        if msg.get("content"):
                            msg["content"] = strip_thinking(msg["content"])
                    # Prepend session start notice or memory warning
                    mem_warning, avail_gb = get_memory_warning()
                    first_msg = resp_data.get("choices", [{}])[0].get("message", {})
                    orig = first_msg.get("content") or ""
                    if is_session_start:
                        first_msg["content"] = make_session_start_notice(avail_gb) + orig
                    elif mem_warning:
                        first_msg["content"] = f"{mem_warning}\n\n{orig}"
                    # Alias rewrite
                    resp_str = json.dumps(resp_data)
                    for path, alias in ALIAS_REVERSE.items():
                        resp_str = resp_str.replace(path, alias)
                    resp_bytes = resp_str.encode()
                except Exception as e:
                    log(f"RESPONSE_PARSE_ERROR {e}")

                resp_headers = {k: v for k, v in resp.headers.items()
                                if k.lower() in ("content-type",)}
                if resp.status != 200:
                    log(f"CHAT_BACKEND_NON200 status={resp.status} body={resp_bytes[:300]!r}")
                else:
                    log(f"CHAT_BACKEND_OK status={resp.status}")
                return web.Response(status=resp.status, body=resp_bytes, headers=resp_headers)

    except Exception as e:
        log(f"BACKEND_ERROR {e}")
        return web.Response(status=502, body=json.dumps({"error": str(e)}).encode())
    finally:
        _inflight -= 1
        log(f"INFLIGHT_DEC count={_inflight}")


async def main():
    log("=== Async proxy started (SSE heartbeat + tool-calling) ===")
    app = web.Application()
    app.router.add_route("*", "/{path_info:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    log(f"Listening on 0.0.0.0:{PORT} → default={DEFAULT_BACKEND} aliases={list(MODEL_ALIASES.keys())}")
    print(f"MLX-VLM async proxy on 0.0.0.0:{PORT} → default={DEFAULT_BACKEND}")
    print(f"Aliases: {list(MODEL_ALIASES.keys())}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
