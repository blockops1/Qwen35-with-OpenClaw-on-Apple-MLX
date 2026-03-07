"""
proxy.py — Async proxy for vllm-mlx (Qwen3.5 on Apple MLX)

Provides full OpenAI API compatibility with:
  - Model alias rewriting (friendly name → local model path)
  - Tool calling: Qwen-Agent XML prompt injection + <tool_call> response parsing
  - SSE heartbeat keepalives (prevents socket timeouts during long prefills)
  - Thinking token suppression (<think>…</think> and 'Thinking Process:' stripped)
  - Structured request/response logging

Port 8080 (proxy) → forwards to vllm-mlx on port 8091

Configure MODEL_ALIASES and LOG_FILE below before running.
"""
import asyncio
import json
import os
import re
import uuid
import datetime
import aiohttp
from aiohttp import web

BACKEND = "http://127.0.0.1:8091"              # vllm-mlx port
PORT = 8080                                    # proxy listen port
LOG_FILE = os.path.expanduser("~/mlx-server/proxy.log")  # persistent log path
HEARTBEAT_INTERVAL = 5                         # seconds between SSE keepalive chunks

# Add your model aliases here — friendly name → full path to MLX model directory
MODEL_ALIASES = {
    "qwen35": "/Users/yourname/mlx-models/Qwen3.5-27B-4bit",
    # "qwen9b": "/Users/yourname/mlx-models/Qwen3.5-9B-Instruct-4bit",
}
ALIAS_REVERSE = {v: k for k, v in MODEL_ALIASES.items()}

# Request size limits — prevent runaway sessions from hanging vllm-mlx
# A 42K-token request takes 300s+ to process, causing a timeout that locks
# the model and kills all concurrent sessions. Fail fast with 413 instead.
MAX_INPUT_TOKENS = 35000   # estimated from message content (~4 chars/token)
MAX_MESSAGES    = 300      # runaway session guard

# Concurrency limit — prevent request pile-up from saturating vllm-mlx KV cache.
# Even legitimately-sized requests exhaust KV cache if too many run concurrently.
# With a large OpenClaw system prompt (~18-20K tokens), each request consumes
# ~1GB of KV cache. Tune MAX_CONCURRENT to fit your hardware:
#   16GB + 9B model  → 1
#   32GB + 9B model  → 2-3
#   64GB + 27B model → 1 (recommended); 2 if system prompts are small
#   64GB + 9B model  → 3-4
# A 429 response triggers OpenClaw's model fallback (e.g. to Claude) automatically.
MAX_CONCURRENT  = 1        # max simultaneous in-flight backend requests
_inflight       = 0        # current in-flight count (asyncio single-thread safe)

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


async def fetch_backend_blocking(body_bytes: bytes, headers: dict, path_qs: str):
    """Non-streaming backend fetch. Returns (bytes, status_int)."""
    url = BACKEND + path_qs
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method="POST",
                url=url,
                headers=headers,
                data=body_bytes,
                timeout=aiohttp.ClientTimeout(total=600),
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


async def handle_tool_stream(request: web.Request, data: dict, headers: dict, body_bytes: bytes):
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

    # Launch backend request concurrently
    backend_task = asyncio.create_task(
        fetch_backend_blocking(body_bytes, headers, request.path_qs)
    )

    # Send heartbeat chunks until backend responds
    try:
        while not backend_task.done():
            heartbeat = make_sse_chunk(request_id, model_alias, {"content": ""})
            if not await safe_write(response, heartbeat, "heartbeat"):
                break  # client gone, but let backend_task finish so we can log it
            try:
                await asyncio.wait_for(asyncio.shield(backend_task), timeout=float(HEARTBEAT_INTERVAL))
                break  # backend done
            except asyncio.TimeoutError:
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
        f"body_len={len(body_bytes)} model={requested_model} is_chat={is_chat}")

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
        url = BACKEND + request.path_qs
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

    # Concurrency guard — reject immediately if too many requests in flight
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
                        "Start a new session."
                    ),
                    "type": "request_too_large",
                    "code": 413,
                }
            }).encode()
        )

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
            return await handle_tool_stream(request, data, headers, body_bytes)
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
    url = BACKEND + request.path_qs

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
    log(f"Listening on 0.0.0.0:{PORT} → {BACKEND}")
    print(f"MLX-VLM async proxy on 0.0.0.0:{PORT} → {BACKEND}")
    print(f"Aliases: {list(MODEL_ALIASES.keys())}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
