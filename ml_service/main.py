from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml_service.api.routes import router
from ml_service.config import settings


app = FastAPI(
    title=settings.service_name,
    version="0.1.0",
    description="Cascaded anti-deepfake verification ML service.",
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
