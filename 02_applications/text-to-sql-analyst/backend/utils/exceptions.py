"""
backend/utils/exceptions.py
---------------------------
Custom exception hierarchy. FastAPI exception handlers map each to a
clean JSON response so the frontend can render meaningful errors.
"""
from typing import Any


class AppException(Exception):
    """Base for all application errors."""
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class UnsafeQueryError(AppException):
    status_code = 400
    code = "unsafe_query"


class SQLGenerationError(AppException):
    status_code = 422
    code = "sql_generation_failed"


class SQLExecutionError(AppException):
    status_code = 500
    code = "sql_execution_failed"


class AuthError(AppException):
    status_code = 401
    code = "unauthorized"


class RateLimitError(AppException):
    status_code = 429
    code = "rate_limited"
