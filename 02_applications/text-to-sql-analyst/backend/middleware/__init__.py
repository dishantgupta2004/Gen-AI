"""
backend/middleware/__init__.py
------------------------------
Two pieces of middleware:
  - RequestIDMiddleware: attaches an X-Request-ID for tracing.
  - RateLimitMiddleware: simple token-bucket per remote IP (in-memory).
                         For production use, swap for slowapi+Redis.
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_per_minute: int = 30) -> None:
        super().__init__(app)
        self._max = max_per_minute
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/health") or request.url.path == "/":
            return await call_next(request)
        key = (request.client.host if request.client else "anon") + ":" + (
            request.headers.get("authorization", "")[-12:] or "noauth"
        )
        now = time.time()
        bucket = self._buckets[key]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= self._max:
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limited", "detail": "Too many requests"},
            )
        bucket.append(now)
        return await call_next(request)
