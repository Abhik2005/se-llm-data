"""
api/main.py — SE-LLM API Server (FastAPI).

OpenAI-compatible API so any tool that uses OpenAI can switch to SE-LLM
with just a base_url change.

Endpoints:
    POST /v1/chat/completions   — Main chat endpoint
    POST /v1/completions        — Raw completion (code autocomplete)
    GET  /v1/models             — List available models
    GET  /v1/usage              — User's usage stats
    POST /auth/register         — Register and get API key
    GET  /health                — Health check

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Deploy to Railway/Render:
    Set environment variable: MODEL_CHECKPOINT_PATH=/path/to/sft_final.pt
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import torch
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.database import get_db, init_db, User, APIKey, UsageLog, TIER_LIMITS
from api.auth import validate_api_key, register_user
from api.rate_limit import rate_limiter, check_token_limit
from model.config import ModelConfig
from model.transformer import Transformer
from evaluation.generate import load_model_from_checkpoint, load_tokenizer, chat_turn, generate_completion

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Global model state ────────────────────────────────────────────────────────

class ModelState:
    model:     Optional[Transformer] = None
    tokenizer: Optional[object]      = None
    device:    torch.device          = torch.device("cpu")
    model_id:  str                   = "se-llm-350m"

state = ModelState()


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    logger.info("Starting SE-LLM API server...")

    # Initialize database
    init_db()

    # Load model
    ckpt_path  = os.getenv("MODEL_CHECKPOINT_PATH", "checkpoints_sft/sft_final.pt")
    tok_path   = os.getenv("TOKENIZER_PATH", "tokenizer/tokenizer.json")

    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found at {ckpt_path} — API will return errors until model is loaded")
    else:
        logger.info(f"Loading model from {ckpt_path}...")
        state.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state.model, _  = load_model_from_checkpoint(ckpt_path, state.device)
        state.tokenizer = load_tokenizer(tok_path)
        logger.info(f"Model loaded on {state.device}")

    yield

    logger.info("Shutting down SE-LLM API server")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SE-LLM API",
    description="Software Engineering AI — OpenAI-compatible API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (allow all origins for now, restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role:    str
    content: str


class ChatCompletionRequest(BaseModel):
    model:       str              = "se-llm-350m"
    messages:    list[ChatMessage]
    max_tokens:  int              = Field(default=512, ge=1, le=2048)
    temperature: float            = Field(default=0.7, ge=0.0, le=2.0)
    top_p:       float            = Field(default=0.95, ge=0.0, le=1.0)
    top_k:       int              = Field(default=50, ge=0)
    stream:      bool             = False   # Streaming not yet implemented


class CompletionRequest(BaseModel):
    model:       str   = "se-llm-350m"
    prompt:      str
    max_tokens:  int   = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p:       float = Field(default=0.95, ge=0.0, le=1.0)
    top_k:       int   = Field(default=50, ge=0)


class RegisterRequest(BaseModel):
    email: str
    name:  Optional[str] = None


# ── Helper ────────────────────────────────────────────────────────────────────

def require_model():
    """Raise 503 if model is not loaded."""
    if state.model is None or state.tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is starting up — try again in a moment.",
        )


def log_usage(db, user_id, api_key_id, endpoint, prompt_tokens, completion_tokens, latency_ms):
    """Record API usage to database."""
    log = UsageLog(
        user_id           = user_id,
        api_key_id        = api_key_id,
        endpoint          = endpoint,
        prompt_tokens     = prompt_tokens,
        completion_tokens = completion_tokens,
        total_tokens      = prompt_tokens + completion_tokens,
        latency_ms        = latency_ms,
    )
    db.add(log)
    db.commit()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": state.model is not None,
        "device":       str(state.device),
        "model_id":     state.model_id,
    }


@app.post("/auth/register")
async def register(req: RegisterRequest, db=Depends(get_db)):
    """
    Register a new user and receive an API key.
    The key is shown ONCE — save it immediately.
    """
    user, raw_key = register_user(req.email, req.name or "", db)
    return {
        "message":  "Registration successful. Save your API key — it won't be shown again.",
        "email":    user.email,
        "tier":     user.tier,
        "api_key":  raw_key,
        "usage": {
            "requests_per_day": TIER_LIMITS["free"]["requests_per_day"],
            "max_tokens":       TIER_LIMITS["free"]["max_tokens"],
        },
    }


@app.get("/v1/models")
async def list_models(auth=Depends(validate_api_key)):
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id":       "se-llm-350m",
                "object":   "model",
                "created":  1720000000,
                "owned_by": "se-llm",
            }
        ],
    }


@app.get("/v1/usage")
async def get_usage(auth=Depends(validate_api_key), db=Depends(get_db)):
    """Return current usage stats for the authenticated user."""
    user, api_key = auth
    remaining = rate_limiter.get_remaining(user.id, user.tier)
    limits     = TIER_LIMITS.get(user.tier, TIER_LIMITS["free"])

    return {
        "tier":        user.tier,
        "usage":       remaining,
        "limits":      limits,
        "upgrade_url": "/pricing" if user.tier == "free" else None,
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    auth=Depends(validate_api_key),
    db=Depends(get_db),
):
    """
    OpenAI-compatible chat completions endpoint.
    Drop-in replacement for openai.ChatCompletion.create()
    """
    user, api_key = auth
    require_model()

    # Rate limiting
    rate_limiter.check_and_increment(user.id, user.tier)

    # Token limit
    max_tokens = check_token_limit(req.max_tokens, user.tier)

    # Extract system prompt and conversation history
    system_prompt = "You are SE-LLM, an expert software engineering assistant."
    history       = []
    user_message  = ""

    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_message = msg.content
        elif msg.role == "assistant":
            history.append({"role": "assistant", "content": msg.content})

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in messages")

    # Generate response
    t0 = time.time()
    try:
        response_text = chat_turn(
            state.model,
            state.tokenizer,
            user_message,
            system_prompt=system_prompt,
            history=history,
            max_new_tokens=max_tokens,
            temperature=req.temperature,
            device=state.device,
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail="Model generation failed")

    latency_ms = (time.time() - t0) * 1000

    # Estimate token counts
    prompt_tokens     = sum(len(m.content.split()) for m in req.messages) * 4 // 3
    completion_tokens = len(response_text.split()) * 4 // 3

    # Log usage
    log_usage(db, user.id, api_key.id, "chat/completions",
              prompt_tokens, completion_tokens, latency_ms)

    # OpenAI-compatible response format
    return {
        "id":      f"chatcmpl-{int(time.time())}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   "se-llm-350m",
        "choices": [
            {
                "index":         0,
                "message":       {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/completions")
async def completions(
    req: CompletionRequest,
    auth=Depends(validate_api_key),
    db=Depends(get_db),
):
    """
    Raw completion endpoint — ideal for code autocomplete.
    Completes the given prompt without chat formatting.
    """
    user, api_key = auth
    require_model()

    rate_limiter.check_and_increment(user.id, user.tier)
    max_tokens = check_token_limit(req.max_tokens, user.tier)

    t0 = time.time()
    try:
        completion = generate_completion(
            state.model,
            state.tokenizer,
            req.prompt,
            max_new_tokens=max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            device=state.device,
        )
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail="Model generation failed")

    latency_ms        = (time.time() - t0) * 1000
    prompt_tokens     = len(req.prompt.split()) * 4 // 3
    completion_tokens = len(completion.split()) * 4 // 3

    log_usage(db, user.id, api_key.id, "completions",
              prompt_tokens, completion_tokens, latency_ms)

    return {
        "id":      f"cmpl-{int(time.time())}",
        "object":  "text_completion",
        "created": int(time.time()),
        "model":   "se-llm-350m",
        "choices": [
            {
                "text":          completion,
                "index":         0,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      prompt_tokens + completion_tokens,
        },
    }


# ── Run locally ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
