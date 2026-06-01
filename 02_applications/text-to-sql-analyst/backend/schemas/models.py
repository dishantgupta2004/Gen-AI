"""
backend/schemas/models.py
-------------------------
Pydantic v2 request/response models.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---- Auth ----

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---- Query ----

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    session_id: str = Field(..., min_length=4, max_length=128)
    include_chart: bool = True


class QueryResponse(BaseModel):
    sql: str
    explanation: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    duration_ms: int
    chart: dict[str, Any] | None = None


# ---- Export ----

class ExportRequest(BaseModel):
    sql: str
    fmt: str = Field("csv", pattern="^(csv|xlsx)$")
    filename: str = "results"


# ---- Schema ----

class SchemaResponse(BaseModel):
    schema_text: str
    table_count: int
