import logging
import time
import uuid

from fastapi import FastAPI, Request

from app.config import settings
from app.routers import api
from app.services.groq_service import groq_service
from app.services.rate_limiter import rate_limiter, MODEL_RATE_LIMITS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Groq API Server",
    description="Groq LLM API proxy with OpenAI-compatible endpoints, rate limit tracking, and Bearer auth",
    version="1.0.0",
)

# Include routers
app.include_router(api.router)
app.include_router(api.openai_router)


# CORS middleware (off by default)
if settings.enable_cors:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS enabled for origins: %s", settings.cors_origins)


# Health check (no auth)
@app.get("/health")
async def health():
    has_key = bool(settings.groq_api_key and settings.groq_api_key != "")
    models = list(MODEL_RATE_LIMITS.keys())
    return {
        "status": "ok" if has_key else "degraded",
        "groq_api_configured": has_key,
        "default_model": settings.default_model,
        "max_concurrent": settings.max_concurrent,
        "available_models": models,
        "rate_limits_summary": {
            m: {"rpm": l["rpm"], "tpm": l["tpm"]}
            for m, l in MODEL_RATE_LIMITS.items()
        },
    }


# Request logging middleware with request ID tracking
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    start = time.monotonic()
    response = await call_next(request)
    duration = int((time.monotonic() - start) * 1000)
    logger.info(
        "[%s] %s %s -> %d (%dms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.on_event("shutdown")
async def shutdown_event():
    await groq_service.close()
