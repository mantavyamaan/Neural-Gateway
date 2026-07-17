"""
Neural Gateway service entrypoint.

Run locally with:
    uvicorn app.main:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as neural_gateway_gateway
from app.config import CORS_ORIGINS, ROUTER_VERSION

from contextlib import asynccontextmanager

from app.core.database import init_db
from app.core.openrouter_sync import sync_openrouter_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    
    # Upsert local built-in registry (Generative models, etc.)
    from app.models.registry_builder import MODEL_REGISTRY
    from app.core.database import bulk_upsert_models
    for m in MODEL_REGISTRY:
        m["evidence"]["eligible_for_auto_route"] = True
    bulk_upsert_models(MODEL_REGISTRY)
    
    # Pull live models from OpenRouter (only seeds if DB is empty)
    sync_openrouter_models()
    # Note: benchmark_sync is called automatically inside sync_openrouter_models()
    # but run it independently on startup to handle cases where OpenRouter is unreachable
    try:
        from app.core.benchmark_sync import run_benchmark_sync
        run_benchmark_sync()
    except Exception:
        pass  # non-critical -- routing still works with existing DB data

    # Warmup the Embedding Semantic Parser so the ONNX session is loaded in memory
    try:
        from app.core.embedding_parser import get_parser
        logging.getLogger("uvicorn.error").info("Warming up embedding semantic parser (ONNX)...")
        _ = get_parser().parse("warmup prompt")
        logging.getLogger("uvicorn.error").info("Parser warmed up successfully.")
    except Exception as e:
        logging.getLogger("uvicorn.error").error(f"Failed to warmup embedding parser: {e}")
        
    yield

app = FastAPI(
    title="Neural Gateway",
    description=(
        "Adaptive Task and LLM Allocation System — a complete, end-to-end AI agent "
        "and neural routing gateway. Neural Gateway decides which model should handle a request "
        "using Bayesian inference, and acts as an intelligent proxy to actively generate "
        "and stream the final response back to the user."
    ),
    version=ROUTER_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,  # required so browsers forward x-openrouter-key and x-neural_gateway-admin-key headers
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(neural_gateway_gateway, tags=["neural_gateway-gateway"])


@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "Neural Gateway",
        "version": ROUTER_VERSION,
        "docs": "/docs",
    }
