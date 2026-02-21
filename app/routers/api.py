import json
import logging
import time
import uuid
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.auth import verify_token
from app.config import settings
from app.schemas import ChatRequest, GroqApiResponse, OpenAIChatRequest
from app.services.groq_service import groq_service, MODEL_MAP, OPENAI_MODEL_ALIASES
from app.services.rate_limiter import rate_limiter, MODEL_RATE_LIMITS
from app.services.usage_tracker import usage_tracker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["api"])
openai_router = APIRouter(tags=["openai-compatible"])


# --- Cached model list builders ---

# The three actual Groq models
GROQ_MODELS = [
    {"id": "openai/gpt-oss-120b", "short": "gpt-oss-120b"},
    {"id": "openai/gpt-oss-20b", "short": "gpt-oss-20b"},
    {"id": "llama-3.3-70b-versatile", "short": "llama-70b"},
    {"id": "qwen/qwen3-32b", "short": "qwen-32b"},
]


@lru_cache(maxsize=1)
def _build_openai_models_list() -> dict:
    """Build the OpenAI-compatible models list (cached, static data)."""
    models = []
    seen_ids = set()

    for m in GROQ_MODELS:
        models.append({
            "id": m["id"],
            "object": "model",
            "created": 1700000000,
            "owned_by": "groq",
        })
        seen_ids.add(m["id"])

    # Add OpenAI aliases
    for alias, resolved_id in OPENAI_MODEL_ALIASES.items():
        if alias not in seen_ids:
            models.append({
                "id": alias,
                "object": "model",
                "created": 1700000000,
                "owned_by": "groq",
                "parent": resolved_id,
            })
            seen_ids.add(alias)

    return {"object": "list", "data": models}


# --- Custom REST API endpoints ---

@router.post("/chat", response_model=GroqApiResponse)
async def chat(req: ChatRequest, _: str = Depends(verify_token)):
    """General-purpose AI chat via Groq. Send a prompt, get a response."""
    resp = await groq_service.chat(
        prompt=req.prompt,
        model=req.model,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    if resp.is_error:
        raise HTTPException(status_code=502, detail=resp.result)
    return resp.to_dict()


@router.get("/models")
async def list_models(_: str = Depends(verify_token)):
    """List available Groq models with rate limit info."""
    models = []
    for m in GROQ_MODELS:
        limits = MODEL_RATE_LIMITS.get(m["id"], {})
        models.append({
            "key": m["short"],
            "model_id": m["id"],
            "default": m["id"] == settings.default_model,
            "rate_limits": limits,
        })
    return {"models": models}


@router.get("/rate-limits")
async def get_rate_limits(_: str = Depends(verify_token)):
    """Get current rate limit usage for all models."""
    return {"rate_limits": rate_limiter.get_all_status()}


@router.get("/usage/summary")
async def usage_summary(hours: int = 24, _: str = Depends(verify_token)):
    """Get usage summary for the last N hours (default 24)."""
    return usage_tracker.get_summary(hours=hours)


@router.get("/usage/recent")
async def usage_recent(limit: int = 50, _: str = Depends(verify_token)):
    """Get recent usage log entries."""
    return {"entries": usage_tracker.get_recent(limit=limit)}


@router.get("/usage/hourly")
async def usage_hourly(hours: int = 24, _: str = Depends(verify_token)):
    """Get hourly aggregated usage stats for charting."""
    return {"hourly": usage_tracker.get_hourly_stats(hours=hours)}


@router.get("/usage/all-time")
async def usage_all_time(_: str = Depends(verify_token)):
    """Get all-time usage totals."""
    return usage_tracker.get_all_time_stats()


# --- OpenAI-compatible endpoints ---

@openai_router.post("/v1/chat/completions")
async def openai_chat_completions(req: OpenAIChatRequest, _: str = Depends(verify_token)):
    """OpenAI-compatible chat completions endpoint. Proxies to Groq API."""
    _, model_id = groq_service.resolve_model(req.model)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Convert OpenAI messages to plain dicts for Groq
    messages = []
    for msg in req.messages:
        m = {"role": msg.role, "content": msg.text()}
        if msg.name:
            m["name"] = msg.name
        messages.append(m)

    if not any(m["content"] for m in messages if m["role"] == "user"):
        raise HTTPException(status_code=400, detail="No user messages provided")

    # --- Streaming path ---
    if req.stream:
        try:
            stream_gen = await groq_service.chat_completions(
                messages=messages,
                model=req.model,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
                top_p=req.top_p,
                stop=req.stop,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

        return StreamingResponse(
            stream_gen,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --- Non-streaming path ---
    try:
        data, headers, duration = await groq_service.chat_completions(
            messages=messages,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stream=False,
            top_p=req.top_p,
            stop=req.stop,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Groq already returns OpenAI-compatible format, just override the ID
    data["id"] = request_id
    return data


@openai_router.get("/v1/models")
async def openai_list_models(_: str = Depends(verify_token)):
    """OpenAI-compatible model listing."""
    return _build_openai_models_list()
