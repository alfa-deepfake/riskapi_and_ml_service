from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml_service.api.routes import router
from ml_service.config import settings
from ml_service.services.rppg_service import warm_rppg_model


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # The rPPG model takes ~1min to build; warm it so the first pulse
    # request does not stall. No-op when the runtime is not installed.
    threading.Thread(target=warm_rppg_model, daemon=True).start()
    yield


app = FastAPI(
    title=settings.service_name,
    version="0.1.0",
    description="Cascaded anti-deepfake verification ML service.",
    lifespan=_lifespan,
)
origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
