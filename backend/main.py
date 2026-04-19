"""FastAPI entry point for the crack-detection backend.

Run directly with::

    uvicorn backend.main:app --reload --port 8000

or via the provided script::

    python -m backend

The app exposes:

- ``GET  /api/health``
- ``GET  /api/samples`` / ``GET /api/samples/{name}``
- ``POST /api/predict``  (multipart ``file`` or ``?sample=<name>``)
- ``POST /api/gradcam``  (same inputs, returns ``image/png``)
- ``POST /api/segment`` (same inputs, returns overlay + mask + stats)
- ``POST /api/ai/report`` / ``POST /api/ai/chat`` — HKBU GenAI calls
- ``POST /api/evaluate`` / ``GET /api/dataset-image?path=``
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import config
from backend.deps import get_model, get_seg_model
from backend.routers import ai, evaluate, health, predict, samples, segment


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm the model caches so the first predict / segment request is not slow.
    try:
        get_model()
    except Exception as exc:  # pragma: no cover - keep the API up even if load fails
        print(f"[backend] Warning: classifier preload failed ({exc}).")
    try:
        if get_seg_model() is None:
            print(
                "[backend] Note: no segmentation checkpoint at "
                f"{config.SEG_MODEL_PATH}. /api/segment will return 503."
            )
    except Exception as exc:  # pragma: no cover
        print(f"[backend] Warning: segmentation preload failed ({exc}).")
    yield


app = FastAPI(
    title="Crack Detection API",
    version="1.0.0",
    description=(
        "Backend for the concrete crack classifier demo. Owns the PyTorch "
        "model, Grad-CAM computation, and HKBU GenAI integration."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_prefix = "/api"
app.include_router(health.router, prefix=api_prefix)
app.include_router(samples.router, prefix=api_prefix)
app.include_router(predict.router, prefix=api_prefix)
app.include_router(segment.router, prefix=api_prefix)
app.include_router(ai.router, prefix=api_prefix)
app.include_router(evaluate.router, prefix=api_prefix)


@app.get("/")
def root() -> dict:
    return {
        "service": "crack-detection-backend",
        "docs": "/docs",
        "health": f"{api_prefix}/health",
    }
