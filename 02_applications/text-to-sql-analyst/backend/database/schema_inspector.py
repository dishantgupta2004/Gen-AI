"""
backend/database/schema_inspector.py
------------------------------------
Introspects the live PostgreSQL schema and renders a compact,
LLM-friendly description used to ground SQL generation.

Why not pass the whole information_schema to the LLM? Because:
  - Enterprise DBs have thousands of columns -> token explosion.
  - The LLM hallucinates less when given a tight, structured DDL view.

We cache the schema in memory (TTL) and lazily refresh.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool
    is_pk: bool = False
    fk_target: str | None = None  # "schema.table.column"


@dataclass
class TableInfo:
    schema: str
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    row_estimate: int = 0
    comment: str | None = None

    @property
    def fqn(self) -> str:
        return f"{self.schema}.{self.name}"


class SchemaCache:
    def __init__(self, ttl_seconds: int = 600) -> None:
        self._ttl = ttl_seconds
        self._loaded_at: float = 0.0
        self._tables: Dict[str, TableInfo] = {}

    def is_stale(self) -> bool:
        return (time.time() - self._loaded_at) > self._ttl

    def set(self, tables: Dict[str, TableInfo]) -> None:
        self._tables = tables
        self._loaded_at = time.time()

    @property
    def tables(self) -> Dict[str, TableInfo]:
        return self._tables


_cache = SchemaCache(ttl_seconds=600)


# ---- Introspection queries -------------------------------------------------

_TABLES_SQL = """
SELECT n.nspname  AS schema,
       c.relname  AS name,
       c.reltuples::bigint AS row_estimate,
       obj_description(c.oid) AS comment
FROM   pg_class c
JOIN   pg_namespace n ON n.oid = c.relnamespace
WHERE  c.relkind IN ('r','p','v','m')
  AND  n.nspname NOT IN ('pg_catalog','information_schema')
  AND  n.nspname NOT LIKE 'pg_toast%'
ORDER  BY n.nspname, c.relname;
"""

_COLUMNS_SQL = """
SELECT c.table_schema, c.table_name, c.column_name,
       c.data_type, c.is_nullable
FROM   information_schema.columns c
WHERE  c.table_schema NOT IN ('pg_catalog','information_schema')
ORDER  BY c.table_schema, c.table_name, c.ordinal_position;
"""

_PK_SQL = """
SELECT tc.table_schema, tc.table_name, kcu.column_name
FROM   information_schema.table_constraints tc
JOIN   information_schema.key_column_usage kcu
       ON tc.constraint_name = kcu.constraint_name
      AND tc.table_schema    = kcu.table_schema
WHERE  tc.constraint_type = 'PRIMARY KEY';
"""

_FK_SQL = """
SELECT tc.table_schema   AS src_schema,
       tc.table_name     AS src_table,
       kcu.column_name   AS src_column,
       ccu.table_schema  AS tgt_schema,
       ccu.table_name    AS tgt_table,
       ccu.column_name   AS tgt_column
FROM   information_schema.table_constraints tc
JOIN   information_schema.key_column_usage kcu
       ON tc.constraint_name = kcu.constraint_name
JOIN   information_schema.constraint_column_usage ccu
       ON tc.constraint_name = ccu.constraint_name
WHERE  tc.constraint_type = 'FOREIGN KEY';
"""


async def load_schema(session: AsyncSession, force: bool = False) -> Dict[str, TableInfo]:
    """Load (or refresh) the schema cache."""
    if not force and _cache.tables and not _cache.is_stale():
        return _cache.tables

    tables: Dict[str, TableInfo] = {}

    # Tables
    for row in (await session.execute(text(_TABLES_SQL))).mappings():
        t = TableInfo(
            schema=row["schema"],
            name=row["name"],
            row_estimate=int(row["row_estimate"] or 0),
            comment=row["comment"],
        )
        tables[t.fqn] = t

    # Columns
    for row in (await session.execute(text(_COLUMNS_SQL))).mappings():
        fqn = f"{row['table_schema']}.{row['table_name']}"
        if fqn not in tables:
            continue
        tables[fqn].columns.append(
            ColumnInfo(
                name=row["column_name"],
                data_type=row["data_type"],
                nullable=(row["is_nullable"] == "YES"),
            )
        )

    # Primary keys
    pk_lookup = {
        (r["table_schema"], r["table_name"], r["column_name"])
        for r in (await session.execute(text(_PK_SQL))).mappings()
    }
    for t in tables.values():
        for c in t.columns:
            if (t.schema, t.name, c.name) in pk_lookup:
                c.is_pk = True

    # Foreign keys
    for r in (await session.execute(text(_FK_SQL))).mappings():
        fqn = f"{r['src_schema']}.{r['src_table']}"
        if fqn not in tables:
            continue
        for c in tables[fqn].columns:
            if c.name == r["src_column"]:
                c.fk_target = f"{r['tgt_schema']}.{r['tgt_table']}.{r['tgt_column']}"

    _cache.set(tables)
    logger.info("Schema loaded: %d tables", len(tables))
    return tables


def render_schema_for_llm(
    tables: Dict[str, TableInfo],
    allowed_schemas: tuple[str, ...] = ("public",),
    max_tables: int = 80,
) -> str:
    """
    Render schema as compact DDL-like text. We deliberately avoid raw
    pg_dump output because it's noisy. Each table is one block:

        TABLE public.orders  -- 1.2M rows
          id            integer PRIMARY KEY
          customer_id   integer FK -> public.customers.id
          total_amount  numeric
          created_at    timestamp
    """
    lines: list[str] = []
    selected = [
        t for t in tables.values() if t.schema in allowed_schemas
    ][:max_tables]

    for t in selected:
        header = f"TABLE {t.fqn}"
        if t.row_estimate:
            header += f"  -- ~{t.row_estimate:,} rows"
        if t.comment:
            header += f"  -- {t.comment}"
        lines.append(header)
        for c in t.columns:
            parts = [f"  {c.name:24}", c.data_type]
            if c.is_pk:
                parts.append("PRIMARY KEY")
            if c.fk_target:
                parts.append(f"FK -> {c.fk_target}")
            if not c.nullable and not c.is_pk:
                parts.append("NOT NULL")
            lines.append(" ".join(parts))
        lines.append("")

    return "\n".join(lines)
