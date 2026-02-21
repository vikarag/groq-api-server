#!/usr/bin/env python3
"""Groq API Server — Test Tool

Usage:
    python3 test_server.py                    # Run all tests
    python3 test_server.py --skip-ai          # Skip AI calls (~2s)
    python3 test_server.py --only security    # Run specific category
    python3 test_server.py --url http://remote:8021 --token xxx
    python3 test_server.py --verbose          # Show response bodies
"""

import argparse
import http.client
import json
import os
import sys
import time
from urllib.parse import urlparse


# ── ANSI colors ──────────────────────────────────────────────────

class C:
    PASS = "\033[92m"  # green
    FAIL = "\033[91m"  # red
    SKIP = "\033[93m"  # yellow
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    HEADER = "\033[96m"  # cyan


def _no_color():
    for attr in ("PASS", "FAIL", "SKIP", "BOLD", "DIM", "RESET", "HEADER"):
        setattr(C, attr, "")


# ── HTTP client ──────────────────────────────────────────────────

class HttpClient:
    def __init__(self, base_url: str, token: str | None, timeout: int = 60):
        parsed = urlparse(base_url)
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 8021)
        self.scheme = parsed.scheme or "http"
        self.token = token
        self.timeout = timeout

    def _conn(self) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout)
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        auth: bool = True,
        raw_response: bool = False,
        timeout: int | None = None,
    ) -> dict:
        """Send a request. Returns dict with status, body, json, headers, elapsed_ms."""
        headers = {"Content-Type": "application/json"}
        if auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        conn = self._conn()
        if timeout:
            conn.timeout = timeout

        encoded = json.dumps(body).encode() if body else None
        start = time.monotonic()

        try:
            conn.request(method, path, body=encoded, headers=headers)
            resp = conn.getresponse()
            elapsed = int((time.monotonic() - start) * 1000)

            if raw_response:
                return {"status": resp.status, "response": resp, "elapsed_ms": elapsed, "headers": dict(resp.getheaders())}

            raw_body = resp.read().decode(errors="replace")
            result = {
                "status": resp.status,
                "body": raw_body,
                "json": None,
                "headers": dict(resp.getheaders()),
                "elapsed_ms": elapsed,
            }
            try:
                result["json"] = json.loads(raw_body)
            except (json.JSONDecodeError, ValueError):
                pass
            return result
        except Exception as e:
            elapsed = int((time.monotonic() - start) * 1000)
            return {"status": 0, "body": str(e), "json": None, "headers": {}, "elapsed_ms": elapsed, "error": str(e)}
        finally:
            conn.close()

    def request_sse(self, method: str, path: str, body: dict | None = None) -> dict:
        """Send request expecting SSE response. Returns parsed events."""
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        conn = self._conn()
        encoded = json.dumps(body).encode() if body else None
        start = time.monotonic()

        try:
            conn.request(method, path, body=encoded, headers=headers)
            resp = conn.getresponse()
            elapsed_first = int((time.monotonic() - start) * 1000)

            if resp.status != 200:
                raw = resp.read().decode(errors="replace")
                return {"status": resp.status, "events": [], "body": raw, "elapsed_ms": elapsed_first}

            events = []
            done = False
            raw_data = resp.read().decode(errors="replace")
            elapsed = int((time.monotonic() - start) * 1000)

            for line in raw_data.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        done = True
                        events.append({"type": "done"})
                    else:
                        try:
                            events.append({"type": "data", "data": json.loads(payload)})
                        except json.JSONDecodeError:
                            events.append({"type": "raw", "data": payload})

            return {"status": resp.status, "events": events, "done": done, "elapsed_ms": elapsed}
        except Exception as e:
            elapsed = int((time.monotonic() - start) * 1000)
            return {"status": 0, "events": [], "error": str(e), "elapsed_ms": elapsed}
        finally:
            conn.close()


# ── Test runner ──────────────────────────────────────────────────

class TestRunner:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[dict] = []
        self._current_category = ""

    def category(self, name: str):
        self._current_category = name
        print(f"\n{C.HEADER}[{name}]{C.RESET}")

    def _record(self, name: str, status: str, msg: str = "", elapsed: int = 0):
        self.results.append({"category": self._current_category, "name": name, "status": status})
        icon = {"PASS": C.PASS + "PASS" + C.RESET, "FAIL": C.FAIL + "FAIL" + C.RESET, "SKIP": C.SKIP + "SKIP" + C.RESET}[status]
        time_str = f" ({elapsed}ms)" if elapsed else ""
        detail = f" — {msg}" if msg else ""
        print(f"  {icon} {name}{time_str}{detail}")

    def passed(self, name: str, msg: str = "", elapsed: int = 0):
        self._record(name, "PASS", msg, elapsed)

    def failed(self, name: str, msg: str = "", elapsed: int = 0):
        self._record(name, "FAIL", msg, elapsed)

    def skipped(self, name: str, msg: str = ""):
        self._record(name, "SKIP", msg)

    def log_verbose(self, label: str, data):
        if self.verbose:
            if isinstance(data, dict):
                text = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                text = str(data)
            if len(text) > 2000:
                text = text[:2000] + f"\n... ({len(text)} chars total)"
            print(f"    {C.DIM}{label}: {text}{C.RESET}")

    def summary(self) -> int:
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        skipped = sum(1 for r in self.results if r["status"] == "SKIP")

        print(f"\n{'=' * 50}")
        parts = []
        if passed:
            parts.append(f"{C.PASS}{passed} passed{C.RESET}")
        if failed:
            parts.append(f"{C.FAIL}{failed} failed{C.RESET}")
        if skipped:
            parts.append(f"{C.SKIP}{skipped} skipped{C.RESET}")
        print(f"  Results: {', '.join(parts)}  ({total} total)")
        print(f"{'=' * 50}")

        return 0 if failed == 0 else 1


# ── Token loader ─────────────────────────────────────────────────

def load_token(args_token: str | None) -> str | None:
    if args_token:
        return args_token
    env_token = os.environ.get("API_SECRET_TOKEN")
    if env_token:
        return env_token
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("API_SECRET_TOKEN=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ── Test categories ──────────────────────────────────────────────

def run_connectivity_tests(client: HttpClient, runner: TestRunner):
    runner.category("Connectivity")

    # 1. Health check
    r = client.request("GET", "/health", auth=False)
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 200 and r.get("json", {}).get("status") in ("ok", "degraded"):
        j = r["json"]
        runner.passed("test_health", f"status={j['status']}, model={j.get('default_model')}", r["elapsed_ms"])
    else:
        runner.failed("test_health", f"status={r['status']}, body={r['body'][:100]}", r["elapsed_ms"])

    # 2. Response time
    elapsed = r["elapsed_ms"]
    if elapsed < 1000:
        runner.passed("test_response_time", f"{elapsed}ms < 1000ms", elapsed)
    else:
        runner.failed("test_response_time", f"{elapsed}ms >= 1000ms", elapsed)

    # 3. Health includes models
    models = r.get("json", {}).get("available_models", [])
    if len(models) == 3:
        runner.passed("test_health_models", f"{len(models)} models listed")
    else:
        runner.failed("test_health_models", f"expected 3 models, got {len(models)}")


def run_auth_tests(client: HttpClient, runner: TestRunner):
    runner.category("Authentication")

    # 1. Valid token -> 200
    r = client.request("GET", "/api/models", auth=True)
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 200:
        runner.passed("test_auth_valid_token", "", r["elapsed_ms"])
    else:
        runner.failed("test_auth_valid_token", f"expected 200, got {r['status']}", r["elapsed_ms"])

    # 2. No token -> 401 or 403
    r = client.request("GET", "/api/models", auth=False)
    runner.log_verbose("response", r.get("json"))
    if r["status"] in (401, 403):
        runner.passed("test_auth_no_token", f"{r['status']} as expected", r["elapsed_ms"])
    else:
        runner.failed("test_auth_no_token", f"expected 401/403, got {r['status']}", r["elapsed_ms"])

    # 3. Wrong token
    saved_token = client.token
    client.token = "wrong-token-12345"
    r = client.request("GET", "/api/models", auth=True)
    client.token = saved_token
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 401:
        runner.passed("test_auth_invalid_token", "401 as expected", r["elapsed_ms"])
    else:
        runner.failed("test_auth_invalid_token", f"expected 401, got {r['status']}", r["elapsed_ms"])


def run_api_tests(client: HttpClient, runner: TestRunner, skip_ai: bool):
    runner.category("Custom REST API")

    # 1. List models
    r = client.request("GET", "/api/models")
    runner.log_verbose("response", r.get("json"))
    models = r.get("json", {}).get("models", [])
    if r["status"] == 200 and len(models) == 3:
        keys = [m["key"] for m in models]
        runner.passed("test_api_models", f"3 models: {', '.join(keys)}", r["elapsed_ms"])
    else:
        runner.failed("test_api_models", f"expected 3 models, got {len(models)}", r["elapsed_ms"])

    # 2. Models have rate limit info
    if models and "rate_limits" in models[0]:
        runner.passed("test_api_models_rate_limits", "rate_limits present in model info")
    else:
        runner.failed("test_api_models_rate_limits", "rate_limits missing from model info")

    # 3. Rate limits endpoint
    r = client.request("GET", "/api/rate-limits")
    runner.log_verbose("response", r.get("json"))
    rate_limits = r.get("json", {}).get("rate_limits", {})
    if r["status"] == 200 and len(rate_limits) == 3:
        runner.passed("test_api_rate_limits", f"{len(rate_limits)} models tracked", r["elapsed_ms"])
    else:
        runner.failed("test_api_rate_limits", f"expected 3 models, got {len(rate_limits)}", r["elapsed_ms"])

    # 4. Chat (AI)
    if skip_ai:
        runner.skipped("test_api_chat", "--skip-ai")
    else:
        r = client.request("POST", "/api/chat", {"prompt": "Say only: hello", "model": "llama-3.3-70b-versatile"}, timeout=120)
        runner.log_verbose("response", r.get("json"))
        if r["status"] == 200 and r.get("json", {}).get("result"):
            result_len = len(r["json"]["result"])
            runner.passed("test_api_chat", f"{result_len} chars", r["elapsed_ms"])
        else:
            runner.failed("test_api_chat", f"status={r['status']}, body={r['body'][:200]}", r["elapsed_ms"])

    # 5. Validation — empty prompt
    r = client.request("POST", "/api/chat", {"prompt": ""})
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 422:
        runner.passed("test_api_validation", "422 on empty prompt", r["elapsed_ms"])
    else:
        runner.failed("test_api_validation", f"expected 422, got {r['status']}", r["elapsed_ms"])


def run_openai_tests(client: HttpClient, runner: TestRunner, skip_ai: bool):
    runner.category("OpenAI-Compatible API")

    # 1. List models
    r = client.request("GET", "/v1/models")
    runner.log_verbose("response", r.get("json"))
    models_data = r.get("json", {}).get("data", [])
    if r["status"] == 200 and len(models_data) > 0:
        model_ids = [m["id"] for m in models_data]
        runner.passed("test_openai_models", f"{len(models_data)} models", r["elapsed_ms"])
    else:
        runner.failed("test_openai_models", f"status={r['status']}", r["elapsed_ms"])
        model_ids = []

    # 2. gpt-4o alias exists
    if "gpt-4o" in model_ids:
        runner.passed("test_openai_gpt4o_alias", "gpt-4o present")
    elif model_ids:
        runner.failed("test_openai_gpt4o_alias", f"gpt-4o not in {model_ids}")
    else:
        runner.skipped("test_openai_gpt4o_alias", "models list empty")

    # 3. All three real models present
    real_models = {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"}
    found = real_models.intersection(set(model_ids))
    if len(found) == 3:
        runner.passed("test_openai_real_models", "all 3 Groq models present")
    else:
        runner.failed("test_openai_real_models", f"found {len(found)}/3: {found}")

    # 4. Chat completions non-streaming (AI)
    if skip_ai:
        runner.skipped("test_openai_chat", "--skip-ai")
    else:
        body = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Say only: ok"}],
            "stream": False,
        }
        r = client.request("POST", "/v1/chat/completions", body, timeout=120)
        runner.log_verbose("response", r.get("json"))
        j = r.get("json", {})
        if (r["status"] == 200
                and j.get("choices", [{}])[0].get("message", {}).get("content")):
            content = j["choices"][0]["message"]["content"]
            runner.passed("test_openai_chat", f"{len(content)} chars", r["elapsed_ms"])
        else:
            runner.failed("test_openai_chat", f"status={r['status']}, body={r['body'][:200]}", r["elapsed_ms"])

    # 5. Chat completions with gpt-4o alias (AI)
    if skip_ai:
        runner.skipped("test_openai_chat_alias", "--skip-ai")
    else:
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Say only: test"}],
            "stream": False,
        }
        r = client.request("POST", "/v1/chat/completions", body, timeout=120)
        runner.log_verbose("response", r.get("json"))
        j = r.get("json", {})
        if r["status"] == 200 and j.get("choices"):
            actual_model = j.get("model", "")
            runner.passed("test_openai_chat_alias", f"gpt-4o -> {actual_model}", r["elapsed_ms"])
        else:
            runner.failed("test_openai_chat_alias", f"status={r['status']}", r["elapsed_ms"])

    # 6. Chat completions streaming (AI)
    if skip_ai:
        runner.skipped("test_openai_chat_stream", "--skip-ai")
    else:
        body = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Say only: ok"}],
            "stream": True,
        }
        r = client.request_sse("POST", "/v1/chat/completions", body)
        runner.log_verbose("sse_events", f"{len(r.get('events', []))} events, done={r.get('done')}")
        events = r.get("events", [])
        has_done = r.get("done", False)

        has_role = False
        has_content = False
        has_finish = False
        for ev in events:
            if ev.get("type") == "data":
                d = ev["data"]
                choices = d.get("choices", [{}])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "role" in delta:
                        has_role = True
                    if "content" in delta:
                        has_content = True
                    fr = choices[0].get("finish_reason")
                    if fr and fr != "null":
                        has_finish = True

        if r["status"] == 200 and has_done and (has_content or has_role):
            detail_parts = [f"{len(events)} events"]
            if has_role:
                detail_parts.append("role")
            if has_content:
                detail_parts.append("content")
            if has_finish:
                detail_parts.append("finish_reason")
            detail_parts.append("[DONE]")
            runner.passed("test_openai_chat_stream", ", ".join(detail_parts), r["elapsed_ms"])
        else:
            runner.failed("test_openai_chat_stream",
                          f"status={r['status']}, events={len(events)}, done={has_done}, content={has_content}",
                          r.get("elapsed_ms", 0))

    # 7. Validation — empty messages
    r = client.request("POST", "/v1/chat/completions", {"model": "llama-3.3-70b-versatile", "messages": []})
    runner.log_verbose("response", r.get("json"))
    if r["status"] == 422:
        runner.passed("test_openai_validation", "422 on empty messages", r["elapsed_ms"])
    else:
        runner.failed("test_openai_validation", f"expected 422, got {r['status']}", r["elapsed_ms"])


def run_rate_limit_tests(client: HttpClient, runner: TestRunner):
    runner.category("Rate Limits")

    # 1. Rate limits endpoint returns data for all 3 models
    r = client.request("GET", "/api/rate-limits")
    runner.log_verbose("response", r.get("json"))
    rate_limits = r.get("json", {}).get("rate_limits", {})

    expected_models = {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"}
    found_models = set(rate_limits.keys())
    if expected_models == found_models:
        runner.passed("test_rate_limits_all_models", "all 3 models tracked")
    else:
        runner.failed("test_rate_limits_all_models", f"expected {expected_models}, got {found_models}")

    # 2. Each model has correct structure
    all_valid = True
    for model, data in rate_limits.items():
        required_keys = {"requests_per_minute", "requests_per_day", "tokens_per_minute", "tokens_per_day"}
        if not required_keys.issubset(set(data.keys())):
            all_valid = False
            runner.failed("test_rate_limits_structure", f"{model} missing keys: {required_keys - set(data.keys())}")
            break

    if all_valid:
        runner.passed("test_rate_limits_structure", "all models have correct structure")

    # 3. Limits match expected values
    llama_limits = rate_limits.get("llama-3.3-70b-versatile", {})
    rpm_limit = llama_limits.get("requests_per_minute", {}).get("limit", 0)
    tpm_limit = llama_limits.get("tokens_per_minute", {}).get("limit", 0)
    if rpm_limit == 30 and tpm_limit == 12000:
        runner.passed("test_rate_limits_values", f"llama-70b: RPM={rpm_limit}, TPM={tpm_limit}")
    else:
        runner.failed("test_rate_limits_values", f"expected RPM=30,TPM=12000, got RPM={rpm_limit},TPM={tpm_limit}")


# ── Main ─────────────────────────────────────────────────────────

CATEGORIES = {
    "connectivity": run_connectivity_tests,
    "auth": run_auth_tests,
    "api": run_api_tests,
    "openai": run_openai_tests,
    "rate_limits": run_rate_limit_tests,
}

# Categories that need skip_ai parameter
_AI_CATEGORIES = {"api", "openai"}


def main():
    parser = argparse.ArgumentParser(description="Groq API Server Test Tool")
    parser.add_argument("--url", default="http://localhost:8021", help="Server URL (default: http://localhost:8021)")
    parser.add_argument("--token", default=None, help="API token (default: from .env)")
    parser.add_argument("--skip-ai", action="store_true", help="Skip tests that call AI models")
    parser.add_argument("--only", choices=list(CATEGORIES.keys()), help="Run only this category")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show response bodies")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        _no_color()

    token = load_token(args.token)
    if not token:
        print(f"{C.FAIL}Error: No API token found. Use --token, set API_SECRET_TOKEN, or create .env{C.RESET}")
        sys.exit(2)

    client = HttpClient(args.url, token)
    runner = TestRunner(verbose=args.verbose)

    print(f"{C.BOLD}Groq API Server Test Tool{C.RESET}")
    print(f"Server: {args.url}")
    if args.skip_ai:
        print(f"{C.DIM}Mode: --skip-ai (AI calls excluded){C.RESET}")

    # Connectivity pre-check
    try:
        r = client.request("GET", "/health", auth=False, timeout=5)
        if r["status"] == 0:
            print(f"\n{C.FAIL}Error: Cannot connect to {args.url} — {r.get('error', 'unknown')}{C.RESET}")
            sys.exit(2)
    except Exception as e:
        print(f"\n{C.FAIL}Error: Cannot connect to {args.url} — {e}{C.RESET}")
        sys.exit(2)

    categories_to_run = [args.only] if args.only else list(CATEGORIES.keys())

    for cat_name in categories_to_run:
        func = CATEGORIES[cat_name]
        if cat_name in _AI_CATEGORIES:
            func(client, runner, args.skip_ai)
        else:
            func(client, runner)

    exit_code = runner.summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
