# Groq API Server - Task Report

**Date**: 2026-02-21
**Project**: groq-api-server
**Repository**: https://github.com/vikarag/groq-api-server (public)
**Live URL**: https://groq.dhammastack.com

---

## Overview

Built a complete FastAPI-based API proxy for Groq LLM models, mirroring the architecture of the existing `claude-api-server` but using Groq's HTTP API via httpx. The server provides OpenAI-compatible endpoints, per-model rate limit tracking, Bearer token authentication, a web chat UI, a CLI client, and a usage dashboard.

---

## Tasks Completed

### 1. Planning & Design

- Explored existing `claude-api-server` codebase to understand architecture patterns
- Fetched Groq rate limit documentation from https://console.groq.com/docs/rate-limits
- Created implementation plan at `.omc/plans/groq-api-server-plan.md`
- Plan approved by user before execution

### 2. Core Server Implementation (16 files)

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app, CORS, health endpoint, request logging middleware |
| `app/config.py` | Pydantic Settings from `.env` |
| `app/auth.py` | Bearer token authentication |
| `app/schemas.py` | Pydantic models (ChatRequest, OpenAI-compatible schemas, rate limit schemas) |
| `app/routers/api.py` | All API endpoints (custom REST + OpenAI-compatible) |
| `app/services/groq_service.py` | httpx AsyncClient to Groq API, model resolution, SSE streaming |
| `app/services/rate_limiter.py` | Sliding window rate limiter (per-model RPM/RPD/TPM/TPD) |
| `.env` | Groq API key + bearer token (gitignored) |
| `.env.example` | Template with placeholder values |
| `requirements.txt` | fastapi, uvicorn, pydantic-settings, python-dotenv, httpx |
| `test_server.py` | 21 tests across 5 categories |
| `groq-api-server.service` | systemd service unit |
| `.gitignore` | Excludes .env, venv, __pycache__, .claude, .omc |
| `README.md` | Full documentation |

### 3. Models Configured

| Model | Short Name | RPM | TPM | TPD |
|-------|-----------|-----|-----|-----|
| `openai/gpt-oss-120b` | `gpt-oss-120b` | 30 | 8,000 | 200,000 |
| `openai/gpt-oss-20b` | `gpt-oss-20b` | 30 | 8,000 | 200,000 |
| `llama-3.3-70b-versatile` | `llama-70b` | 30 | 12,000 | 100,000 |

**OpenAI Aliases**: `gpt-4o` -> 120b, `gpt-4o-mini` -> 20b, `gpt-3.5-turbo` -> llama-70b

### 4. API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Server status + model info |
| `/api/chat` | POST | Yes | Custom chat endpoint |
| `/api/models` | GET | Yes | List models with rate limits |
| `/api/rate-limits` | GET | Yes | Current rate limit usage |
| `/v1/chat/completions` | POST | Yes | OpenAI-compatible (stream + non-stream) |
| `/v1/models` | GET | Yes | OpenAI-compatible model list |
| `/` | GET | No | Web Chat UI |
| `/dashboard` | GET | No | Usage Dashboard |

### 5. Rate Limiting (3-Layer Approach)

1. **Pre-flight check**: Sliding window counters (RPM/RPD/TPM/TPD per model) checked before each request
2. **Groq header tracking**: `x-ratelimit-remaining-requests` and `x-ratelimit-remaining-tokens` headers extracted and stored
3. **429 handling**: Groq 429 responses caught and returned with clear error messages

### 6. Testing

- **Skip-AI tests**: 17 passed (connectivity, auth, API structure, validation, rate limits)
- **Full tests with Groq API**: 21/21 passed (including live chat, streaming, model aliases)
- Test categories: connectivity, authentication, custom API, OpenAI-compatible, rate limits

### 7. Deployment

- **systemd service**: Installed at `/etc/systemd/system/groq-api-server.service`, enabled for boot
- **Port**: 8021 (localhost only)
- **Caddy reverse proxy**: `http://groq.dhammastack.com` -> `localhost:8021`
- **Fix**: Group in service file changed from `gslee` to `users` (gslee's primary group is `users`, not `gslee`)

### 8. Caddy Configuration

Added to `/etc/caddy/Caddyfile`:
```
http://groq.dhammastack.com {
    reverse_proxy localhost:8021
    encode gzip
}
```

**Key insight**: Uses `http://` prefix because `dhammastack.com` routes through Cloudflare, which terminates TLS and connects to origin over HTTP port 80. Without `http://`, Caddy would try to get its own certificate, which conflicts with Cloudflare's proxy.

### 9. GitHub Repository

- Created as `vikarag/cc-yjaedan-groq-api-server` (private)
- Renamed to `vikarag/groq-api-server` (public)
- Verified no secrets in commits (`.env` excluded by `.gitignore`)
- 3 commits total

### 10. Web Chat UI (`/`)

- Dark theme, mobile-responsive
- Model selection dropdown (Llama 70B, GPT-OSS 120B, GPT-OSS 20B)
- SSE streaming responses
- Bearer token authentication (stored in localStorage)
- **System prompt panel**: Toggle via header button, persisted in localStorage
- **Conversation export**: JSON, Markdown, or plain text download
- Dashboard link in header

### 11. CLI Chat Tool (`cli_chat.py`)

- Interactive mode with streaming
- One-shot mode (`--one-shot "question"`)
- Model switching (`/model llama`, `/model 120b`)
- System prompt (`/system You are...`)
- Rate limit display (`/rate`)
- Env var configuration (`GROQ_API_TOKEN`, `GROQ_SERVER_URL`, `GROQ_MODEL`)

### 12. Usage Dashboard (`/dashboard`)

- Real-time rate limit cards per model (RPM/RPD/TPM/TPD with color-coded progress bars)
- Canvas-based live usage chart (30 snapshots at 10-second intervals)
- Request log table showing model, RPM used, TPM used, Groq remaining
- Server status overview (health status, model count, total requests/tokens)
- Auto-refresh every 10 seconds

---

## Configuration

| Setting | Value |
|---------|-------|
| Groq API Key | (stored in `.env`, not committed) |
| API Bearer Token | (stored in `.env`, not committed) |
| Port | 8021 |
| Default Model | `llama-3.3-70b-versatile` |
| Max Concurrent | 5 |
| CORS | Enabled (`*`) |

---

## Architecture

```
Browser/CLI -> Caddy (groq.dhammastack.com) -> FastAPI (127.0.0.1:8021)
                                                    |
                                          Bearer Token Auth
                                                    |
                                    +----- Rate Limit Check (pre-flight) -----+
                                    |                                          |
                                    v                                          v
                              httpx AsyncClient                          429 / reject
                                    |
                              Groq API (api.groq.com/openai/v1)
                                    |
                              Response + Rate Limit Headers
                                    |
                              Update sliding windows
                                    |
                              Return to client
```

---

## Files Changed in Caddy

- `/etc/caddy/Caddyfile`: Added `http://groq.dhammastack.com` block (lines 207-210)

---

## Git Commits

1. `0a8adbb` - Initial release: Groq API Server with OpenAI-compatible endpoints
2. `23a87f1` - Add web chat UI, CLI client, and fix systemd service group
3. `aa1c3cd` - Add system prompt, conversation export, and usage dashboard

---

## Issues Encountered & Resolved

1. **systemd Group error (status 216)**: Service file had `Group=gslee` but the user's primary group is `users`. Fixed to `Group=users`.

2. **Caddy "Subdomain not configured" error**: Initially used `groq.dhammastack.com { }` (HTTPS), but Cloudflare terminates TLS and connects to origin via HTTP, hitting the wildcard `http://*.dhammastack.com` catch-all. Fixed by using `http://groq.dhammastack.com { }`.

3. **Stale autopilot state**: Session-scoped state file at `.omc/state/sessions/{id}/autopilot-state.json` wasn't found by `state_clear` tool. Required manual `rm -f` to clean up.

---

## SDK Compatibility

The server works with:
- **OpenAI Python SDK**: `openai.OpenAI(base_url="https://groq.dhammastack.com/v1", api_key="TOKEN")`
- **LangChain**: `ChatOpenAI(base_url="...", api_key="...", model="llama-3.3-70b-versatile")`
- **Any OpenAI-compatible client**: Standard `/v1/chat/completions` and `/v1/models` endpoints
