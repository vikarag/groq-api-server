# Groq API Server

A FastAPI-based API proxy for **Groq LLM models** with OpenAI-compatible endpoints, per-model rate limit tracking, and Bearer token authentication.

Mirrors the architecture of [claude-api-server](../claude-api-server/) but uses Groq's HTTP API instead of Claude CLI.

## Available Models

| Model | Short Name | RPM | TPM | Best For |
|-------|-----------|-----|-----|----------|
| `openai/gpt-oss-120b` | `gpt-oss-120b` | 30 | 8K | Complex reasoning |
| `openai/gpt-oss-20b` | `gpt-oss-20b` | 30 | 8K | Fast general tasks |
| `llama-3.3-70b-versatile` | `llama-70b` | 30 | 12K | Balanced (default) |

### OpenAI Compatibility Aliases

| OpenAI Name | Maps To |
|-------------|---------|
| `gpt-4o` | `openai/gpt-oss-120b` |
| `gpt-4o-mini` | `openai/gpt-oss-20b` |
| `gpt-3.5-turbo` | `llama-3.3-70b-versatile` |

## Quick Start

```bash
# 1. Clone and enter directory
cd groq-api-server

# 2. Create virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env: set GROQ_API_KEY and API_SECRET_TOKEN

# 5. Start server
uvicorn app.main:app --host 127.0.0.1 --port 8021

# 6. Test
curl http://localhost:8021/health
```

## Configuration (.env)

| Setting | Default | Description |
|---------|---------|-------------|
| `GROQ_API_KEY` | (required) | Your Groq API key |
| `API_SECRET_TOKEN` | (required) | Bearer token for client authentication |
| `DEFAULT_MODEL` | `llama-3.3-70b-versatile` | Default model when none specified |
| `MAX_CONCURRENT` | `5` | Max simultaneous requests |
| `REQUEST_TIMEOUT` | `120` | Timeout per request (seconds) |
| `ENABLE_CORS` | `false` | Enable CORS middleware |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |

## Endpoints

### Health Check (No Auth)

```bash
curl http://localhost:8021/health
```

### Custom REST API (Auth Required)

```bash
# Chat
curl -X POST http://localhost:8021/api/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain REST APIs", "model": "llama-3.3-70b-versatile"}'

# List models
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8021/api/models

# Rate limit status
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8021/api/rate-limits
```

### OpenAI-Compatible API (Auth Required)

```bash
# Chat completions
curl -X POST http://localhost:8021/v1/chat/completions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl -X POST http://localhost:8021/v1/chat/completions \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# List models
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8021/v1/models
```

## SDK Integration

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8021/v1",
    api_key="YOUR_TOKEN"
)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8021/v1",
    api_key="YOUR_TOKEN",
    model="llama-3.3-70b-versatile"
)

response = llm.invoke("What is AI?")
```

## Rate Limits (Groq Free Tier)

| Model | RPM | RPD | TPM | TPD |
|-------|-----|-----|-----|-----|
| openai/gpt-oss-120b | 30 | 1,000 | 8,000 | 200,000 |
| openai/gpt-oss-20b | 30 | 1,000 | 8,000 | 200,000 |
| llama-3.3-70b-versatile | 30 | 1,000 | 12,000 | 100,000 |

Rate limits are tracked locally and visible at `/api/rate-limits`. The server pre-checks limits before sending requests to Groq to avoid wasting round-trips.

## Testing

```bash
# Quick tests (no AI calls)
python3 test_server.py --skip-ai

# Full tests (calls Groq API)
python3 test_server.py

# Specific category
python3 test_server.py --only auth

# Verbose output
python3 test_server.py --verbose
```

## Deployment (systemd)

```bash
sudo cp groq-api-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable groq-api-server
sudo systemctl start groq-api-server
```

## Architecture

```
Client -> Bearer Auth -> FastAPI (8021) -> httpx -> Groq API (api.groq.com)
                                                 |
                                    Rate limit tracking (per-model)
                                                 |
                                    Response -> Client (with rate info)
```
