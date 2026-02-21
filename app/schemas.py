"""Request/Response Pydantic schemas for Groq API Server."""

from typing import Any, Union

from pydantic import BaseModel, Field


# --- Custom REST API schemas ---

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100000)
    model: str | None = Field(None, description="Model: gpt-oss-120b, gpt-oss-20b, llama-70b")
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class GroqApiResponse(BaseModel):
    result: str
    model: str
    usage: dict
    duration_ms: int
    is_error: bool = False


# --- OpenAI-compatible schemas ---

class OpenAIMessage(BaseModel):
    model_config = {"extra": "allow"}
    role: str  # "system", "user", "assistant"
    content: Union[str, list[Any], None] = None
    name: str | None = None

    def text(self) -> str:
        """Extract plain text from content (handles str and array-of-parts)."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts = []
        for part in self.content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)


class OpenAIChatRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str = "llama-3.3-70b-versatile"
    messages: list[OpenAIMessage] = Field(..., min_length=1)
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    top_p: float | None = None
    stop: Union[str, list[str], None] = None


class OpenAIChatChoice(BaseModel):
    model_config = {"extra": "allow"}
    index: int = 0
    message: dict
    finish_reason: str = "stop"


class OpenAIChatResponse(BaseModel):
    model_config = {"extra": "allow"}
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatChoice]
    usage: dict


# --- Rate limit schemas ---

class ModelRateLimitInfo(BaseModel):
    model_id: str
    rpm_limit: int
    rpd_limit: int
    tpm_limit: int
    tpd_limit: int
    rpm_remaining: int | None = None
    rpd_remaining: int | None = None
    tpm_remaining: int | None = None
    requests_in_window: int = 0
    tokens_in_window: int = 0
