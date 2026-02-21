"""Groq API client using httpx for async HTTP calls."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import httpx

from app.config import settings
from app.services.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)

# Model map: short name -> full Groq model ID
MODEL_MAP = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "llama-70b": "llama-3.3-70b-versatile",
    # Full IDs map to themselves
    "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
}

# OpenAI-compatible model name aliases -> Groq model IDs
OPENAI_MODEL_ALIASES = {
    "gpt-4o": "openai/gpt-oss-120b",
    "gpt-4": "openai/gpt-oss-120b",
    "gpt-4-turbo": "openai/gpt-oss-120b",
    "gpt-4o-mini": "openai/gpt-oss-20b",
    "gpt-3.5-turbo": "llama-3.3-70b-versatile",
}


@dataclass
class GroqResponse:
    result: str
    model: str
    usage: dict
    duration_ms: int
    is_error: bool = False
    rate_limit_headers: dict | None = None

    def to_dict(self) -> dict:
        return {
            "result": self.result,
            "model": self.model,
            "usage": self.usage,
            "duration_ms": self.duration_ms,
            "is_error": self.is_error,
        }


class GroqService:
    def __init__(self):
        self._semaphore = asyncio.Semaphore(settings.max_concurrent)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=settings.groq_base_url,
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(settings.request_timeout, connect=10.0),
            )
        return self._client

    @staticmethod
    def resolve_model(model_name: str | None) -> tuple[str, str]:
        """Resolve any model name (OpenAI alias, short name, or full ID) to (display_name, groq_model_id)."""
        if not model_name:
            model_id = settings.default_model
        else:
            # Check OpenAI aliases first
            model_id = OPENAI_MODEL_ALIASES.get(model_name)
            if not model_id:
                # Check model map (short names and full IDs)
                model_id = MODEL_MAP.get(model_name, model_name)
        return model_id, model_id

    async def chat(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GroqResponse:
        """Send a chat completion request to Groq API."""
        _, model_id = self.resolve_model(model)
        start = time.monotonic()

        # Pre-check rate limits
        allowed, reason = rate_limiter.check_limit(model_id)
        if not allowed:
            duration = int((time.monotonic() - start) * 1000)
            return GroqResponse(
                result=reason,
                model=model_id,
                usage={},
                duration_ms=duration,
                is_error=True,
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {"model": model_id, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with self._semaphore:
            rate_limiter.record_request(model_id)
            logger.info("Groq API request: model=%s prompt_len=%d", model_id, len(prompt))
            try:
                client = await self._get_client()
                resp = await client.post("/chat/completions", json=payload)
            except httpx.TimeoutException:
                duration = int((time.monotonic() - start) * 1000)
                return GroqResponse(
                    result=f"Request timed out after {settings.request_timeout}s",
                    model=model_id,
                    usage={},
                    duration_ms=duration,
                    is_error=True,
                )
            except httpx.HTTPError as e:
                duration = int((time.monotonic() - start) * 1000)
                return GroqResponse(
                    result=f"HTTP error: {e}",
                    model=model_id,
                    usage={},
                    duration_ms=duration,
                    is_error=True,
                )

        duration = int((time.monotonic() - start) * 1000)

        # Update rate limiter from response headers
        rate_limiter.update_from_headers(model_id, dict(resp.headers))

        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after", "unknown")
            return GroqResponse(
                result=f"Rate limited by Groq. Retry after {retry_after}s",
                model=model_id,
                usage={},
                duration_ms=duration,
                is_error=True,
                rate_limit_headers=dict(resp.headers),
            )

        if resp.status_code != 200:
            try:
                error_data = resp.json()
                error_msg = error_data.get("error", {}).get("message", resp.text)
            except Exception:
                error_msg = resp.text
            return GroqResponse(
                result=f"Groq API error ({resp.status_code}): {error_msg}",
                model=model_id,
                usage={},
                duration_ms=duration,
                is_error=True,
            )

        data = resp.json()
        choices = data.get("choices", [])
        result_text = choices[0]["message"]["content"] if choices else ""

        usage = data.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        if total_tokens:
            rate_limiter.record_tokens(model_id, total_tokens)

        return GroqResponse(
            result=result_text,
            model=data.get("model", model_id),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            duration_ms=duration,
            rate_limit_headers=dict(resp.headers),
        )

    async def chat_completions(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ):
        """Send an OpenAI-compatible chat completions request to Groq.
        If stream=True, returns an async generator of SSE chunks.
        If stream=False, returns the full response dict.
        """
        _, model_id = self.resolve_model(model)
        start = time.monotonic()

        # Pre-check rate limits
        allowed, reason = rate_limiter.check_limit(model_id)
        if not allowed:
            raise Exception(reason)

        payload: dict = {"model": model_id, "messages": messages, "stream": stream}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop

        async with self._semaphore:
            rate_limiter.record_request(model_id)
            logger.info("Groq API completions: model=%s stream=%s msgs=%d", model_id, stream, len(messages))

            try:
                client = await self._get_client()

                if stream:
                    return self._stream_response(client, payload, model_id, start)

                resp = await client.post("/chat/completions", json=payload)
            except httpx.TimeoutException:
                raise Exception(f"Request timed out after {settings.request_timeout}s")
            except httpx.HTTPError as e:
                raise Exception(f"HTTP error: {e}")

        duration = int((time.monotonic() - start) * 1000)
        rate_limiter.update_from_headers(model_id, dict(resp.headers))

        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after", "unknown")
            raise Exception(f"Rate limited by Groq. Retry after {retry_after}s")

        if resp.status_code != 200:
            try:
                error_data = resp.json()
                error_msg = error_data.get("error", {}).get("message", resp.text)
            except Exception:
                error_msg = resp.text
            raise Exception(f"Groq API error ({resp.status_code}): {error_msg}")

        data = resp.json()
        usage = data.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        if total_tokens:
            rate_limiter.record_tokens(model_id, total_tokens)

        return data, dict(resp.headers), duration

    async def _stream_response(self, client: httpx.AsyncClient, payload: dict, model_id: str, start: float):
        """Async generator that yields SSE lines from Groq streaming response."""
        try:
            async with client.stream("POST", "/chat/completions", json=payload) as resp:
                rate_limiter.update_from_headers(model_id, dict(resp.headers))

                if resp.status_code == 429:
                    retry_after = resp.headers.get("retry-after", "unknown")
                    yield f'data: {json.dumps({"error": {"message": f"Rate limited. Retry after {retry_after}s"}})}\n\n'
                    yield "data: [DONE]\n\n"
                    return

                if resp.status_code != 200:
                    raw = await resp.aread()
                    try:
                        error_data = json.loads(raw)
                        error_msg = error_data.get("error", {}).get("message", raw.decode() if isinstance(raw, bytes) else str(raw))
                    except Exception:
                        error_msg = raw.decode() if isinstance(raw, bytes) else str(raw)
                    yield f'data: {json.dumps({"error": {"message": f"Groq API error ({resp.status_code}): {error_msg}"}})}\n\n'
                    yield "data: [DONE]\n\n"
                    return

                total_tokens = 0
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        payload_str = line[6:]
                        if payload_str == "[DONE]":
                            if total_tokens:
                                rate_limiter.record_tokens(model_id, total_tokens)
                            yield "data: [DONE]\n\n"
                            return
                        try:
                            chunk_data = json.loads(payload_str)
                            # Track tokens from usage in final chunk
                            if "usage" in chunk_data:
                                total_tokens = chunk_data["usage"].get("total_tokens", 0)
                            yield f"data: {payload_str}\n\n"
                        except json.JSONDecodeError:
                            yield f"data: {payload_str}\n\n"

        except httpx.TimeoutException:
            yield f'data: {json.dumps({"error": {"message": f"Stream timed out after {settings.request_timeout}s"}})}\n\n'
            yield "data: [DONE]\n\n"
        except httpx.HTTPError as e:
            yield f'data: {json.dumps({"error": {"message": f"HTTP error: {e}"}})}\n\n'
            yield "data: [DONE]\n\n"

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Singleton
groq_service = GroqService()
