"""
tests/test_api.py
-----------------
API-level integration test using FastAPI's TestClient. The agent and
database calls are monkey-patched so we don't need a live Postgres or
Groq key during CI.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest


# Set minimal env BEFORE importing backend.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://test:test@localhost:5432/test",
)
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET", "test-secret-min-32-bytes-12345678")
os.environ.setdefault("DEMO_USER", "analyst")
os.environ.setdefault("DEMO_PASSWORD", "analyst")


@pytest.fixture
def client():
    # Build a fake AsyncSession that satisfies our dependency signature.
    from fastapi.testclient import TestClient

    from backend.database.connection import get_session
    from backend.main import app

    async def _fake_session():
        yield AsyncMock()

    app.dependency_overrides[get_session] = _fake_session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _login(client) -> str:
    r = client.post(
        "/api/v1/auth/login",
        data={"username": "analyst", "password": "analyst"},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "app" in r.json()


def test_login_success(client):
    token = _login(client)
    assert isinstance(token, str) and len(token) > 20


def test_login_bad_credentials(client):
    r = client.post(
        "/api/v1/auth/login",
        data={"username": "analyst", "password": "wrong"},
    )
    assert r.status_code == 401


def test_query_requires_auth(client):
    r = client.post(
        "/api/v1/query",
        json={"question": "show revenue", "session_id": "s-123"},
    )
    assert r.status_code == 401


def test_query_happy_path(client):
    token = _login(client)
    fake_agent_result = type(
        "X", (), {"sql": "SELECT 1 AS x LIMIT 100", "explanation": "Returns one."}
    )()
    fake_exec_result = type(
        "Y",
        (),
        {
            "rows": [{"x": 1}],
            "columns": ["x"],
            "row_count": 1,
            "duration_ms": 3,
            "truncated": False,
            "to_dataframe": lambda self=None: __import__("pandas").DataFrame([{"x": 1}]),
        },
    )()

    with patch("backend.routers.query.get_agent") as mock_get_agent, patch(
        "backend.routers.query.execute_sql", new=AsyncMock(return_value=fake_exec_result)
    ):
        mock_get_agent.return_value.run = AsyncMock(return_value=fake_agent_result)

        r = client.post(
            "/api/v1/query",
            json={"question": "show one", "session_id": "s-abc"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sql"].startswith("SELECT")
    assert body["row_count"] == 1
