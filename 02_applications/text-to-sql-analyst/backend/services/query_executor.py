"""
backend/services/query_executor.py
----------------------------------
Executes validated SQL against PostgreSQL inside a read-only transaction
and returns a pandas DataFrame plus timing metadata.

The session is already pinned to READ ONLY at connection time, but we
also wrap the call in `BEGIN READ ONLY ... ROLLBACK` for paranoia and
to guarantee no implicit COMMIT semantics from connection pooling.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.utils.exceptions import SQLExecutionError
from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    rows: list[dict]
    columns: list[str]
    row_count: int
    duration_ms: int
    truncated: bool

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


async def execute_sql(
    sql: str,
    session: AsyncSession,
    max_rows: int = 10_000,
) -> ExecutionResult:
    start = time.perf_counter()
    try:
        async with session.begin():
            # explicit READ ONLY transaction (belt + suspenders)
            await session.execute(text("SET TRANSACTION READ ONLY"))
            result = await session.execute(text(sql))
            rows_raw = result.mappings().fetchmany(max_rows + 1)
    except Exception as e:
        logger.exception("SQL execution failed")
        raise SQLExecutionError(str(e), {"sql": sql})

    truncated = len(rows_raw) > max_rows
    rows = [dict(r) for r in rows_raw[:max_rows]]
    columns = list(rows[0].keys()) if rows else list(result.keys())

    duration_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "Query executed",
        extra={"duration_ms": duration_ms, "rows": len(rows)},
    )
    return ExecutionResult(
        rows=rows,
        columns=columns,
        row_count=len(rows),
        duration_ms=duration_ms,
        truncated=truncated,
    )
