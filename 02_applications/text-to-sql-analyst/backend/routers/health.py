"""
backend/routers/health.py
"""
from fastapi import APIRouter

from backend.database.connection import healthcheck

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    db_ok = await healthcheck()
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}
