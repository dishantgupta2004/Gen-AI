"""
backend/agents/sql_agent.py
---------------------------
LangGraph-driven Text-to-SQL agent.

GRAPH:

        ┌─────────────┐
        │ load_schema │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │ generate    │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │ validate    │──unsafe──► reject (UnsafeQueryError)
        └──────┬──────┘
               │ safe
               ▼
        ┌─────────────┐
        │  explain    │
        └──────┬──────┘
               ▼
             END

If `validate` flags a fixable parser error, we loop back to `generate`
with the parser feedback (max 1 retry).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.memory import memory
from backend.agents.prompts import (
    build_explain_prompt,
    build_sql_prompt,
)
from backend.config import get_settings
from backend.database.schema_inspector import (
    load_schema,
    render_schema_for_llm,
)
from backend.services.sql_validator import validate_sql
from backend.utils.exceptions import SQLGenerationError, UnsafeQueryError
from backend.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---- LangGraph state -------------------------------------------------------

class AgentState(TypedDict, total=False):
    session_id: str
    question: str
    schema_text: str
    sql: str
    explanation: str
    retries: int
    error: str


@dataclass
class AgentResult:
    sql: str
    explanation: str


# ---- LLM client (Groq Llama 3.x) ------------------------------------------

def _make_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,  # type: ignore[arg-type]
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def _msgs_to_lc(msgs: list[dict[str, str]]):
    """Convert OpenAI-style dicts -> LangChain message objects."""
    out = []
    for m in msgs:
        if m["role"] == "system":
            out.append(SystemMessage(content=m["content"]))
        elif m["role"] == "assistant":
            out.append(AIMessage(content=m["content"]))
        else:
            out.append(HumanMessage(content=m["content"]))
    return out


def _strip_fences(text: str) -> str:
    """Models occasionally wrap SQL in ```sql ... ``` despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.endswith("```"):
            t = t[: -3]
    return t.strip().rstrip(";")


# ---- Nodes ----------------------------------------------------------------

class SQLAgent:
    def __init__(self) -> None:
        self.llm = _make_llm()
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("load_schema", self._load_schema_node)
        g.add_node("generate", self._generate_node)
        g.add_node("validate", self._validate_node)
        g.add_node("explain", self._explain_node)
        g.set_entry_point("load_schema")
        g.add_edge("load_schema", "generate")
        g.add_edge("generate", "validate")
        g.add_conditional_edges(
            "validate",
            lambda s: "retry" if s.get("error") and s.get("retries", 0) < 1 else "ok",
            {"retry": "generate", "ok": "explain"},
        )
        g.add_edge("explain", END)
        return g.compile()

    # ---- node implementations ----

    async def _load_schema_node(self, state: AgentState, *, db: AsyncSession) -> AgentState:
        tables = await load_schema(db)
        state["schema_text"] = render_schema_for_llm(tables)
        return state

    async def _generate_node(self, state: AgentState, **_: Any) -> AgentState:
        history = memory.history(state["session_id"])
        # If retrying, append a self-correction hint.
        question = state["question"]
        if state.get("error"):
            question = (
                f"{state['question']}\n\n"
                f"Your previous SQL failed validation: {state['error']}. "
                f"Produce a corrected single SELECT only."
            )
        msgs = build_sql_prompt(state["schema_text"], question, history)
        resp = await self.llm.ainvoke(_msgs_to_lc(msgs))
        sql = _strip_fences(resp.content if isinstance(resp.content, str) else str(resp.content))
        if not sql:
            raise SQLGenerationError("LLM returned empty SQL")
        state["sql"] = sql
        state["retries"] = state.get("retries", 0) + (1 if state.get("error") else 0)
        state["error"] = ""
        return state

    async def _validate_node(self, state: AgentState, **_: Any) -> AgentState:
        try:
            result = validate_sql(state["sql"], max_rows=settings.max_rows_returned)
            state["sql"] = result.sql
            state["error"] = ""
        except UnsafeQueryError as e:
            # If we're out of retries, propagate; otherwise loop back.
            if state.get("retries", 0) >= 1:
                raise
            state["error"] = e.message
        return state

    async def _explain_node(self, state: AgentState, **_: Any) -> AgentState:
        msgs = build_explain_prompt(state["sql"], state["question"])
        resp = await self.llm.ainvoke(_msgs_to_lc(msgs))
        state["explanation"] = (
            resp.content if isinstance(resp.content, str) else str(resp.content)
        ).strip()
        return state

    # ---- public entrypoint ----

    async def run(self, *, session_id: str, question: str, db: AsyncSession) -> AgentResult:
        # We bind `db` into the schema node via closure since LangGraph
        # nodes don't accept arbitrary kwargs through ainvoke.
        async def _wrapped_load(state: AgentState) -> AgentState:
            return await self._load_schema_node(state, db=db)

        # Rebuild a small graph per-call with the DB-bound first node.
        g = StateGraph(AgentState)
        g.add_node("load_schema", _wrapped_load)
        g.add_node("generate", self._generate_node)
        g.add_node("validate", self._validate_node)
        g.add_node("explain", self._explain_node)
        g.set_entry_point("load_schema")
        g.add_edge("load_schema", "generate")
        g.add_edge("generate", "validate")
        g.add_conditional_edges(
            "validate",
            lambda s: "retry" if s.get("error") and s.get("retries", 0) < 1 else "ok",
            {"retry": "generate", "ok": "explain"},
        )
        g.add_edge("explain", END)
        compiled = g.compile()

        out = await compiled.ainvoke(
            {"session_id": session_id, "question": question, "retries": 0}
        )

        # Persist this turn in conversation memory.
        memory.append(session_id, "user", question)
        memory.append(session_id, "assistant", out["sql"])

        return AgentResult(sql=out["sql"], explanation=out.get("explanation", ""))


_agent_singleton: SQLAgent | None = None


def get_agent() -> SQLAgent:
    global _agent_singleton
    if _agent_singleton is None:
        _agent_singleton = SQLAgent()
    return _agent_singleton
