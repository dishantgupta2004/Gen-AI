"""
frontend/components/api_client.py
---------------------------------
Thin synchronous HTTP client used by the Streamlit app.
"""
from __future__ import annotations

import os
from typing import Any

import requests


class APIClient:
    def __init__(self, base_url: str | None = None, timeout: int = 60) -> None:
        self.base = (base_url or os.getenv("BACKEND_URL", "http://localhost:8000")).rstrip("/")
        self.prefix = "/api/v1"
        self.timeout = timeout
        self.token: str | None = None

    # ---- auth ----
    def login(self, username: str, password: str) -> bool:
        r = requests.post(
            f"{self.base}{self.prefix}/auth/login",
            data={"username": username, "password": password},
            timeout=self.timeout,
        )
        if r.status_code == 200:
            self.token = r.json()["access_token"]
            return True
        return False

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    # ---- API ----
    def query(self, question: str, session_id: str, include_chart: bool = True) -> dict[str, Any]:
        r = requests.post(
            f"{self.base}{self.prefix}/query",
            json={
                "question": question,
                "session_id": session_id,
                "include_chart": include_chart,
            },
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def schema(self) -> dict[str, Any]:
        r = requests.get(
            f"{self.base}{self.prefix}/query/schema",
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def export(self, sql: str, fmt: str = "csv", filename: str = "results") -> bytes:
        r = requests.post(
            f"{self.base}{self.prefix}/query/export",
            json={"sql": sql, "fmt": fmt, "filename": filename},
            headers=self._headers(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.content
