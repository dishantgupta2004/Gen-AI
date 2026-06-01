"""
backend/config.py
-----------------
Centralized application configuration using Pydantic Settings.

All secrets and tunables flow through environment variables (.env), never
hardcoded. Settings are loaded once and cached via lru_cache so the app
imports a single immutable Settings object.
"""
from functools import lru_cache
from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Application ----
    app_name: str = "Text-to-SQL Enterprise Analyst"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # ---- Database (read-only role recommended in production) ----
    # Use a dedicated read-only PostgreSQL role for the agent. Never give
    # the agent connection string write/DDL privileges.
    database_url: str = Field(
        ...,
        description="postgresql+asyncpg://readonly_user:pass@host:5432/db",
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_pool_timeout: int = 30
    statement_timeout_ms: int = 15_000  # Hard server-side query timeout

    # ---- LLM (Groq + Llama 3) ----
    groq_api_key: str = Field(..., description="Groq API key")
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0  # Deterministic SQL generation
    llm_max_tokens: int = 2048

    # ---- Auth ----
    jwt_secret: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # ---- Limits & Safety ----
    max_rows_returned: int = 10_000
    max_query_chars: int = 4_000
    rate_limit_per_minute: int = 30

    # ---- Optional caching ----
    redis_url: str | None = None
    cache_ttl_seconds: int = 300


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
