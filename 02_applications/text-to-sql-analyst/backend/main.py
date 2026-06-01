"""
backend/main.py
---------------
FastAPI entrypoint.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.database.connection import dispose_engine
from backend.middleware import RateLimitMiddleware, RequestIDMiddleware
from backend.routers import auth, health, query
from backend.utils.exceptions import AppException
from backend.utils.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting %s (env=%s)", settings.app_name, settings.environment)
    yield
    await dispose_engine()
    logger.info("Engine disposed; bye.")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)


# Middleware (order matters; outermost runs first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, max_per_minute=settings.rate_limit_per_minute)
app.add_middleware(RequestIDMiddleware)


# Exception handlers
@app.exception_handler(AppException)
async def _app_exc(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "detail": exc.message, "extra": exc.details},
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "Unexpected server error"},
    )


# Routers
app.include_router(health.router)
app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(query.router, prefix=settings.api_v1_prefix)


@app.get("/")
async def root():
    return {"app": settings.app_name, "version": app.version}
