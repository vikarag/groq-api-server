"""Per-model rate limit tracker using sliding window counters."""

import time
from collections import deque
from dataclasses import dataclass, field


# Groq free-tier rate limits per model
MODEL_RATE_LIMITS: dict[str, dict] = {
    "openai/gpt-oss-120b": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    "openai/gpt-oss-20b": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    "llama-3.3-70b-versatile": {"rpm": 30, "rpd": 1000, "tpm": 12000, "tpd": 100000},
}


@dataclass
class ModelWindow:
    """Sliding window tracker for a single model."""
    request_timestamps: deque = field(default_factory=deque)
    daily_request_timestamps: deque = field(default_factory=deque)
    token_entries: deque = field(default_factory=deque)  # (timestamp, token_count)
    daily_token_entries: deque = field(default_factory=deque)

    # Last known values from Groq response headers
    groq_remaining_requests: int | None = None
    groq_remaining_tokens: int | None = None

    def _prune_window(self, dq: deque, window_seconds: float):
        cutoff = time.monotonic() - window_seconds
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _prune_token_window(self, dq: deque, window_seconds: float):
        cutoff = time.monotonic() - window_seconds
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def record_request(self):
        now = time.monotonic()
        self.request_timestamps.append(now)
        self.daily_request_timestamps.append(now)

    def record_tokens(self, count: int):
        now = time.monotonic()
        self.token_entries.append((now, count))
        self.daily_token_entries.append((now, count))

    def requests_in_minute(self) -> int:
        self._prune_window(self.request_timestamps, 60.0)
        return len(self.request_timestamps)

    def requests_in_day(self) -> int:
        self._prune_window(self.daily_request_timestamps, 86400.0)
        return len(self.daily_request_timestamps)

    def tokens_in_minute(self) -> int:
        self._prune_token_window(self.token_entries, 60.0)
        return sum(t[1] for t in self.token_entries)

    def tokens_in_day(self) -> int:
        self._prune_token_window(self.daily_token_entries, 86400.0)
        return sum(t[1] for t in self.daily_token_entries)


class RateLimiter:
    def __init__(self):
        self._windows: dict[str, ModelWindow] = {}

    def _get_window(self, model: str) -> ModelWindow:
        if model not in self._windows:
            self._windows[model] = ModelWindow()
        return self._windows[model]

    def check_limit(self, model: str) -> tuple[bool, str]:
        """Check if a request can proceed. Returns (allowed, reason)."""
        limits = MODEL_RATE_LIMITS.get(model)
        if not limits:
            return True, ""

        window = self._get_window(model)
        rpm = window.requests_in_minute()
        rpd = window.requests_in_day()
        tpm = window.tokens_in_minute()

        if rpm >= limits["rpm"]:
            return False, f"Rate limit exceeded: {rpm}/{limits['rpm']} requests per minute for {model}"
        if rpd >= limits["rpd"]:
            return False, f"Rate limit exceeded: {rpd}/{limits['rpd']} requests per day for {model}"
        if tpm >= limits["tpm"]:
            return False, f"Token limit exceeded: {tpm}/{limits['tpm']} tokens per minute for {model}"

        return True, ""

    def record_request(self, model: str):
        self._get_window(model).record_request()

    def record_tokens(self, model: str, count: int):
        self._get_window(model).record_tokens(count)

    def update_from_headers(self, model: str, headers: dict):
        """Update tracker from Groq response headers."""
        window = self._get_window(model)
        remaining_req = headers.get("x-ratelimit-remaining-requests")
        remaining_tok = headers.get("x-ratelimit-remaining-tokens")
        if remaining_req is not None:
            try:
                window.groq_remaining_requests = int(remaining_req)
            except (ValueError, TypeError):
                pass
        if remaining_tok is not None:
            try:
                window.groq_remaining_tokens = int(remaining_tok)
            except (ValueError, TypeError):
                pass

    def get_status(self, model: str) -> dict:
        """Get current rate limit status for a model."""
        limits = MODEL_RATE_LIMITS.get(model, {"rpm": 0, "rpd": 0, "tpm": 0, "tpd": 0})
        window = self._get_window(model)
        rpm_used = window.requests_in_minute()
        rpd_used = window.requests_in_day()
        tpm_used = window.tokens_in_minute()
        tpd_used = window.tokens_in_day()
        return {
            "model": model,
            "requests_per_minute": {"limit": limits["rpm"], "used": rpm_used, "remaining": limits["rpm"] - rpm_used},
            "requests_per_day": {"limit": limits["rpd"], "used": rpd_used, "remaining": limits["rpd"] - rpd_used},
            "tokens_per_minute": {"limit": limits["tpm"], "used": tpm_used, "remaining": limits["tpm"] - tpm_used},
            "tokens_per_day": {"limit": limits["tpd"], "used": tpd_used, "remaining": limits["tpd"] - tpd_used},
            "groq_remaining_requests": window.groq_remaining_requests,
            "groq_remaining_tokens": window.groq_remaining_tokens,
        }

    def get_all_status(self) -> dict:
        """Get rate limit status for all models."""
        return {model: self.get_status(model) for model in MODEL_RATE_LIMITS}


# Singleton
rate_limiter = RateLimiter()
