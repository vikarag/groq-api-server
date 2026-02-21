#!/usr/bin/env python3
"""Groq Chat CLI - Interactive terminal client for Groq API Server.

Usage:
    python3 cli_chat.py                          # Interactive mode (default model)
    python3 cli_chat.py -m openai/gpt-oss-120b   # Specify model
    python3 cli_chat.py --url https://groq.dhammastack.com  # Custom server URL
    python3 cli_chat.py --one-shot "What is AI?"  # Single question, then exit
"""

import argparse
import json
import os
import sys

try:
    import httpx
except ImportError:
    print("Error: httpx is required. Install with: pip install httpx")
    sys.exit(1)

DEFAULT_URL = os.environ.get("GROQ_SERVER_URL", "http://localhost:8021")
DEFAULT_TOKEN = os.environ.get("GROQ_API_TOKEN", "")
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

MODELS = {
    "1": "llama-3.3-70b-versatile",
    "2": "openai/gpt-oss-120b",
    "3": "openai/gpt-oss-20b",
    "llama": "llama-3.3-70b-versatile",
    "120b": "openai/gpt-oss-120b",
    "20b": "openai/gpt-oss-20b",
}

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "orange": "\033[38;5;208m",
    "blue": "\033[38;5;75m",
    "gray": "\033[38;5;245m",
    "red": "\033[38;5;196m",
    "green": "\033[38;5;82m",
}


def c(color: str, text: str) -> str:
    """Colorize text."""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def print_banner(model: str, url: str):
    print()
    print(c("orange", "  Groq Chat CLI"))
    print(c("dim", f"  Server: {url}"))
    print(c("dim", f"  Model:  {model}"))
    print()
    print(c("gray", "  Commands:"))
    print(c("gray", "    /model <name>  - Switch model (llama, 120b, 20b)"))
    print(c("gray", "    /models        - List available models"))
    print(c("gray", "    /clear         - Clear conversation"))
    print(c("gray", "    /system <msg>  - Set system prompt"))
    print(c("gray", "    /rate          - Show rate limit status"))
    print(c("gray", "    /quit          - Exit"))
    print()


def stream_chat(client: httpx.Client, url: str, token: str,
                messages: list, model: str) -> str:
    """Send chat request with streaming and return full response."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    full_text = ""
    try:
        with client.stream("POST", f"{url}/v1/chat/completions",
                           json=payload, headers=headers, timeout=120) as resp:
            if resp.status_code != 200:
                error_body = resp.read().decode()
                try:
                    detail = json.loads(error_body).get("detail", error_body)
                except json.JSONDecodeError:
                    detail = error_body
                print(c("red", f"\n  Error ({resp.status_code}): {detail}"))
                return ""

            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    parsed = json.loads(data)
                    delta = parsed.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full_text += delta
                except json.JSONDecodeError:
                    pass

    except httpx.ConnectError:
        print(c("red", f"\n  Connection failed: {url}"))
        print(c("gray", "  Is the server running?"))
        return ""
    except httpx.ReadTimeout:
        print(c("red", "\n  Request timed out"))
        return full_text
    except KeyboardInterrupt:
        pass

    print()  # newline after stream
    return full_text


def show_rate_limits(client: httpx.Client, url: str, token: str):
    """Display rate limit status."""
    try:
        resp = client.get(f"{url}/api/rate-limits",
                          headers={"Authorization": f"Bearer {token}"}, timeout=10)
        if resp.status_code != 200:
            print(c("red", f"  Error: {resp.status_code}"))
            return

        data = resp.json().get("rate_limits", {})
        print()
        for model_id, info in data.items():
            rpm_used = info.get("requests_in_window", 0)
            rpm_limit = info.get("rpm_limit", 0)
            tpm_used = info.get("tokens_in_window", 0)
            tpm_limit = info.get("tpm_limit", 0)
            color = "green" if rpm_used < rpm_limit * 0.8 else "orange" if rpm_used < rpm_limit else "red"
            print(f"  {c('bold', model_id)}")
            print(f"    RPM: {c(color, f'{rpm_used}/{rpm_limit}')}  |  TPM: {tpm_used}/{tpm_limit}")
        print()
    except Exception as e:
        print(c("red", f"  Error: {e}"))


def interactive_chat(url: str, token: str, model: str, one_shot: str = None):
    """Main interactive chat loop."""
    messages = []
    system_prompt = None

    with httpx.Client() as client:
        # Verify connection
        try:
            resp = client.get(f"{url}/health", timeout=5)
            if resp.status_code != 200:
                print(c("red", f"Server returned {resp.status_code}"))
                sys.exit(1)
        except httpx.ConnectError:
            print(c("red", f"Cannot connect to {url}"))
            print(c("gray", "Is the server running?"))
            sys.exit(1)

        # Verify token
        resp = client.get(f"{url}/api/models",
                          headers={"Authorization": f"Bearer {token}"}, timeout=5)
        if resp.status_code == 401:
            print(c("red", "Invalid API token"))
            sys.exit(1)

        # One-shot mode
        if one_shot:
            messages = [{"role": "user", "content": one_shot}]
            print(c("blue", f"You: ") + one_shot)
            print()
            print(c("orange", f"[{model}]"))
            stream_chat(client, url, token, messages, model)
            return

        # Interactive mode
        print_banner(model, url)

        while True:
            try:
                user_input = input(c("blue", "You: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print(c("gray", "\n  Bye!"))
                break

            if not user_input:
                continue

            # Commands
            if user_input.startswith("/"):
                cmd = user_input.split(maxsplit=1)
                command = cmd[0].lower()
                arg = cmd[1] if len(cmd) > 1 else ""

                if command in ("/quit", "/exit", "/q"):
                    print(c("gray", "  Bye!"))
                    break
                elif command == "/clear":
                    messages = []
                    print(c("gray", "  Conversation cleared"))
                    continue
                elif command == "/model":
                    resolved = MODELS.get(arg.strip(), arg.strip())
                    model = resolved
                    print(c("gray", f"  Model: {model}"))
                    continue
                elif command == "/models":
                    print(c("gray", "  Available models:"))
                    print(c("gray", "    1) llama-3.3-70b-versatile (llama)"))
                    print(c("gray", "    2) openai/gpt-oss-120b (120b)"))
                    print(c("gray", "    3) openai/gpt-oss-20b (20b)"))
                    continue
                elif command == "/system":
                    system_prompt = arg.strip() if arg.strip() else None
                    if system_prompt:
                        print(c("gray", f"  System prompt set: {system_prompt[:60]}..."))
                    else:
                        print(c("gray", "  System prompt cleared"))
                    continue
                elif command == "/rate":
                    show_rate_limits(client, url, token)
                    continue
                else:
                    print(c("gray", f"  Unknown command: {command}"))
                    continue

            # Build messages
            if system_prompt and not any(m["role"] == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": user_input})

            # Stream response
            print()
            print(c("orange", f"[{model}]"))
            response = stream_chat(client, url, token, messages, model)
            print()

            if response:
                messages.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(description="Groq Chat CLI")
    parser.add_argument("-u", "--url", default=DEFAULT_URL,
                        help=f"Server URL (default: {DEFAULT_URL})")
    parser.add_argument("-t", "--token", default=DEFAULT_TOKEN,
                        help="API bearer token (or set GROQ_API_TOKEN env)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--one-shot", "-1", default=None,
                        help="Single question mode (no interactive loop)")
    args = parser.parse_args()

    if not args.token:
        print(c("orange", "  Groq Chat CLI"))
        print()
        args.token = input("  Enter API token: ").strip()
        if not args.token:
            print(c("red", "  Token required"))
            sys.exit(1)

    interactive_chat(args.url, args.token, args.model, args.one_shot)


if __name__ == "__main__":
    main()
