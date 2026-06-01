"""
backend/mcp_layer/server.py
---------------------------
A FastMCP server that exposes the database to MCP-aware clients
(Claude Desktop, Cursor, our LangGraph agent, etc.).

Why a custom MCP server instead of the off-the-shelf `postgres-mcp`?

  * We can enforce *our* validator (sqlglot + lexical) inside the tool
    boundary. The official server lets any SELECT through; we want our
    rejection logic centralised.
  * We expose the schema as MCP `Resources` so an LLM can lazily fetch
    the description without paying for it on every turn.
  * We can attach business-context Prompts (e.g. "How is revenue defined?")
    as MCP `Prompts`.

Run standalone:
    python -m backend.mcp_layer.server
"""
from __future__ import annotations

import asyncio

from fastmcp import FastMCP

from backend.database.connection import session_scope
from backend.database.schema_inspector import load_schema, render_schema_for_llm
from backend.services.query_executor import execute_sql
from backend.services.sql_validator import validate_sql

mcp = FastMCP("text-to-sql-postgres")


# ---- Resources ------------------------------------------------------------

@mcp.resource("schema://public")
async def schema_resource() -> str:
    """The full public schema, rendered for LLM grounding."""
    async with session_scope() as s:
        tables = await load_schema(s)
    return render_schema_for_llm(tables)


@mcp.resource("schema://tables")
async def tables_list() -> str:
    """Bullet list of available tables."""
    async with session_scope() as s:
        tables = await load_schema(s)
    return "\n".join(f"- {t.fqn} (~{t.row_estimate:,} rows)" for t in tables.values())


# ---- Tools ----------------------------------------------------------------

@mcp.tool()
async def run_select(sql: str, max_rows: int = 1000) -> dict:
    """
    Execute a SELECT statement against the database and return the rows.

    The SQL is validated for safety before execution. Non-SELECT
    statements are rejected.
    """
    safe = validate_sql(sql, max_rows=max_rows)
    async with session_scope() as s:
        result = await execute_sql(safe.sql, s, max_rows=max_rows)
    return {
        "sql": safe.sql,
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "duration_ms": result.duration_ms,
    }


@mcp.tool()
async def describe_table(table: str) -> str:
    """Return DDL-style description of a single table (schema.name)."""
    async with session_scope() as s:
        tables = await load_schema(s)
    info = tables.get(table)
    if not info:
        return f"Table {table!r} not found"
    lines = [f"TABLE {info.fqn}  -- ~{info.row_estimate:,} rows"]
    for c in info.columns:
        flags = []
        if c.is_pk:
            flags.append("PRIMARY KEY")
        if c.fk_target:
            flags.append(f"FK -> {c.fk_target}")
        lines.append(f"  {c.name:24} {c.data_type}  {' '.join(flags)}")
    return "\n".join(lines)


# ---- Prompts --------------------------------------------------------------

@mcp.prompt()
def revenue_definition() -> str:
    """Business context: what 'revenue' means in this database."""
    return (
        "Revenue is defined as SUM(orders.total_amount) for orders with "
        "status = 'completed'. Refunds are tracked separately and should "
        "be subtracted only when explicitly asked for 'net revenue'."
    )


def main() -> None:
    # Default stdio transport (compatible with Claude Desktop & most clients)
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
