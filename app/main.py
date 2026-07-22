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
    import asyncio
    
    # Pull live models from OpenRouter (preserves existing DB priors)
    sync_openrouter_models()
    
    # Run benchmark sync in the background so it doesn't block startup
    async def bg_benchmark_sync():
        try:
            from app.core.benchmark_sync import run_benchmark_sync
            await asyncio.to_thread(run_benchmark_sync)
        except Exception as e:
            logging.getLogger("uvicorn.error").warning(f"Background benchmark sync failed: {e}")
            
    asyncio.create_task(bg_benchmark_sync())

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
