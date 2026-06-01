"""
backend/agents/memory.py
------------------------
Lightweight conversation memory.

For a single-process app this is fine. For a horizontally-scaled
deployment, swap _store with Redis (a Hash per session). The public
interface is identical, so callers don't change.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    ts: float = field(default_factory=time.time)


class ConversationMemory:
    def __init__(self, max_turns_per_session: int = 20, ttl_seconds: int = 60 * 60) -> None:
        self._sessions: Dict[str, Deque[Turn]] = {}
        self._touched: Dict[str, float] = {}
        self._max_turns = max_turns_per_session
        self._ttl = ttl_seconds

    def _gc(self) -> None:
        now = time.time()
        stale = [k for k, t in self._touched.items() if now - t > self._ttl]
        for k in stale:
            self._sessions.pop(k, None)
            self._touched.pop(k, None)

    def append(self, session_id: str, role: str, content: str) -> None:
        self._gc()
        dq = self._sessions.setdefault(session_id, deque(maxlen=self._max_turns))
        dq.append(Turn(role=role, content=content))
        self._touched[session_id] = time.time()

    def history(self, session_id: str) -> list[dict[str, str]]:
        self._gc()
        return [
            {"role": t.role, "content": t.content}
            for t in self._sessions.get(session_id, ())
        ]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._touched.pop(session_id, None)


memory = ConversationMemory()
