#!/usr/bin/env python3
"""
test_proxy.py — vllm-mlx proxy test suite

Tests the full proxy stack: health, chat, thinking suppression,
tool calling, alias rewriting, and SSE parsing.

Usage:
    # Test local stack
    python3 test_proxy.py

    # Test remote machine
    python3 test_proxy.py --host 192.168.1.x

    # Test with a different model alias
    python3 test_proxy.py --model qwen9b

    # Test against a different port
    python3 test_proxy.py --port 9090
"""

import argparse
import json
import sys
import time

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip3 install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# SSE parser
# ---------------------------------------------------------------------------

def parse_response(r):
    """
    Parse either a raw JSON response or an SSE stream response.
    Returns the last meaningful choice dict, or None.
    """
    ct = r.headers.get("content-type", "")
    if "text/event-stream" in ct:
        last = None
        for line in r.text.splitlines():
            if line.startswith("data: ") and "[DONE]" not in line:
                try:
                    chunk = json.loads(line[6:])
                    if chunk.get("choices"):
                        last = chunk
                except Exception:
                    pass
        return last
    try:
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestRunner:
    def __init__(self, base_url: str, model: str):
        self.base = base_url.rstrip("/")
        self.model = model
        self.results = []

    def record(self, name, passed, detail=""):
        self.results.append((name, passed, detail))
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}" + (f"  [{detail}]" if detail else ""))

    def get(self, path, timeout=10):
        return requests.get(f"{self.base}{path}", timeout=timeout)

    def chat(self, messages, max_tokens=100, tools=None, tool_choice=None, timeout=120):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        r = requests.post(
            f"{self.base}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        return r

    def get_content(self, data):
        """Extract text content from parsed response (handles message or delta)."""
        choice = data["choices"][0]
        msg = choice.get("message") or choice.get("delta") or {}
        return msg.get("content") or ""

    def get_tool_calls(self, data):
        """Extract tool_calls from parsed response (handles message or delta)."""
        choice = data["choices"][0]
        msg = choice.get("message") or choice.get("delta") or {}
        return msg.get("tool_calls") or []

    # -----------------------------------------------------------------------
    # Individual tests
    # -----------------------------------------------------------------------

    def test_health(self):
        try:
            r = self.get("/health")
            passed = r.ok and "ok" in r.text.lower() or "healthy" in r.text.lower()
            self.record("Health check", passed, r.text[:60])
        except Exception as e:
            self.record("Health check", False, str(e)[:80])

    def test_models_endpoint(self):
        try:
            r = self.get("/v1/models")
            passed = r.ok
            self.record("/v1/models endpoint", passed, f"HTTP {r.status_code}")
        except Exception as e:
            self.record("/v1/models endpoint", False, str(e)[:80])

    def test_basic_chat(self):
        try:
            t0 = time.time()
            r = self.chat([{"role": "user", "content": "Reply with exactly the word: HELLO"}], max_tokens=20)
            elapsed = time.time() - t0
            if not r.ok:
                self.record("Basic chat", False, f"HTTP {r.status_code}: {r.text[:80]}")
                return
            data = parse_response(r)
            content = self.get_content(data) if data else ""
            passed = bool(content)
            self.record("Basic chat", passed, f"{elapsed:.1f}s — {content[:40]!r}")
        except Exception as e:
            self.record("Basic chat", False, str(e)[:80])

    def test_thinking_stripped(self):
        try:
            r = self.chat([{"role": "user", "content": "What is 2 + 2? Answer with just the number."}], max_tokens=80)
            if not r.ok:
                self.record("Thinking tokens stripped", False, f"HTTP {r.status_code}")
                return
            data = parse_response(r)
            content = self.get_content(data) if data else ""
            has_think = "<think>" in content or "Thinking Process:" in content
            self.record("Thinking tokens stripped", not has_think,
                        "FAIL — thinking leaked" if has_think else f"content={content[:40]!r}")
        except Exception as e:
            self.record("Thinking tokens stripped", False, str(e)[:80])

    def test_no_spurious_tool_call(self):
        """Without tools in request, should never return finish_reason=tool_calls."""
        try:
            r = self.chat([{"role": "user", "content": "What is the capital of France?"}], max_tokens=30)
            if not r.ok:
                self.record("No spurious tool call", False, f"HTTP {r.status_code}")
                return
            data = parse_response(r)
            finish = data["choices"][0].get("finish_reason") if data else "?"
            passed = finish != "tool_calls"
            self.record("No spurious tool call", passed,
                        f"finish_reason={finish}" + (" ← unexpected!" if not passed else ""))
        except Exception as e:
            self.record("No spurious tool call", False, str(e)[:80])

    def test_tool_call_triggered(self):
        """Model should call a tool when one is relevant."""
        tool = [{
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files in a directory on the local filesystem.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Directory path"}},
                    "required": ["path"],
                },
            },
        }]
        try:
            t0 = time.time()
            r = self.chat(
                [{"role": "user", "content": "What files are in /tmp?"}],
                max_tokens=200,
                tools=tool,
                tool_choice="auto",
            )
            elapsed = time.time() - t0
            if not r.ok:
                self.record("Tool call triggered", False, f"HTTP {r.status_code}: {r.text[:80]}")
                return
            data = parse_response(r)
            if not data:
                self.record("Tool call triggered", False, "Empty/unparseable response")
                return
            finish = data["choices"][0].get("finish_reason")
            tool_calls = self.get_tool_calls(data)
            passed = finish == "tool_calls" and len(tool_calls) > 0
            self.record("Tool call triggered", passed,
                        f"{elapsed:.1f}s finish={finish} tools={[tc['function']['name'] for tc in tool_calls]}")
        except Exception as e:
            self.record("Tool call triggered", False, str(e)[:80])

    def test_tool_args_valid_json(self):
        """Tool call arguments must be valid JSON."""
        tool = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }]
        try:
            r = self.chat(
                [{"role": "user", "content": "Read the file /tmp/test.txt"}],
                max_tokens=200,
                tools=tool,
            )
            if not r.ok:
                self.record("Tool args valid JSON", False, f"HTTP {r.status_code}")
                return
            data = parse_response(r)
            if not data:
                self.record("Tool args valid JSON", False, "Empty response")
                return
            tool_calls = self.get_tool_calls(data)
            if not tool_calls:
                finish = data["choices"][0].get("finish_reason")
                self.record("Tool args valid JSON", False,
                            f"No tool_calls in response (finish={finish})")
                return
            raw_args = tool_calls[0]["function"]["arguments"]
            parsed = json.loads(raw_args)
            self.record("Tool args valid JSON", True, f"args={parsed}")
        except json.JSONDecodeError as e:
            self.record("Tool args valid JSON", False, f"Invalid JSON: {raw_args[:60]}")
        except Exception as e:
            self.record("Tool args valid JSON", False, str(e)[:80])

    def test_correct_tool_selected(self):
        """With multiple tools, model should select the relevant one."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from the filesystem.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
        ]
        try:
            r = self.chat(
                [{"role": "user", "content": "What's the weather in Miami?"}],
                max_tokens=200,
                tools=tools,
                tool_choice="auto",
            )
            if not r.ok:
                self.record("Correct tool selected", False, f"HTTP {r.status_code}")
                return
            data = parse_response(r)
            if not data:
                self.record("Correct tool selected", False, "Empty response")
                return
            tool_calls = self.get_tool_calls(data)
            if not tool_calls:
                self.record("Correct tool selected", False, "No tool call made")
                return
            name = tool_calls[0]["function"]["name"]
            passed = name == "get_weather"
            self.record("Correct tool selected", passed,
                        f"called={name}" + (" (wrong!)" if not passed else ""))
        except Exception as e:
            self.record("Correct tool selected", False, str(e)[:80])

    def test_alias_rewrite(self):
        """Response model field should show alias, not full filesystem path."""
        try:
            r = self.chat([{"role": "user", "content": "Hi"}], max_tokens=10)
            if not r.ok:
                self.record("Alias rewrite in response", False, f"HTTP {r.status_code}")
                return
            data = parse_response(r)
            model_field = (data or {}).get("model", "")
            leaked = "/" in model_field and "Users" in model_field
            self.record("Alias rewrite in response", not leaked,
                        f"model={model_field!r}" + (" ← path leaked!" if leaked else ""))
        except Exception as e:
            self.record("Alias rewrite in response", False, str(e)[:80])

    # -----------------------------------------------------------------------
    # Run all
    # -----------------------------------------------------------------------

    def run_all(self):
        print(f"\nTarget: {self.base}  Model alias: {self.model}\n")
        self.test_health()
        self.test_models_endpoint()
        self.test_basic_chat()
        self.test_thinking_stripped()
        self.test_no_spurious_tool_call()
        self.test_tool_call_triggered()
        self.test_tool_args_valid_json()
        self.test_correct_tool_selected()
        self.test_alias_rewrite()

        passed = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        print(f"\n{'='*50}")
        print(f"  {passed}/{total} passed")
        print(f"{'='*50}\n")
        return passed == total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vllm-mlx proxy test suite")
    parser.add_argument("--host", default="localhost", help="Proxy host (default: localhost)")
    parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")
    parser.add_argument("--model", default="qwen35", help="Model alias to test (default: qwen35)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    runner = TestRunner(base_url=base_url, model=args.model)
    success = runner.run_all()
    sys.exit(0 if success else 1)
