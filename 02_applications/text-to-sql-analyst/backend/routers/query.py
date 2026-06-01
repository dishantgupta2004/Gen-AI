"""
backend/routers/query.py
------------------------
The main analytics endpoint. Wires together:

    QueryRequest
      └─ SQLAgent.run() -> sql + explanation        (LLM + validator)
      └─ execute_sql()  -> rows                     (Postgres, RO txn)
      └─ build_chart()  -> Plotly figure JSON       (optional)
      └─ memory.append()                            (multi-turn)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.sql_agent import get_agent
from backend.database.connection import get_session
from backend.database.schema_inspector import load_schema, render_schema_for_llm
from backend.schemas.models import (
    ExportRequest,
    QueryRequest,
    QueryResponse,
    SchemaResponse,
)
from backend.services.export import to_csv_bytes, to_xlsx_bytes
from backend.services.query_executor import execute_sql
from backend.services.sql_validator import validate_sql
from backend.services.visualization import build_chart
from backend.utils.auth import get_current_user
from backend.utils.logging import get_logger

router = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)


@router.post("", response_model=QueryResponse)
async def run_query(
    req: QueryRequest,
    db: AsyncSession = Depends(get_session),
    user=Depends(get_current_user),
) -> QueryResponse:
    agent = get_agent()
    agent_out = await agent.run(session_id=req.session_id, question=req.question, db=db)

    exec_result = await execute_sql(agent_out.sql, db)
    df = exec_result.to_dataframe()

    chart = build_chart(df) if req.include_chart else None

    logger.info(
        "Query completed",
        extra={
            "user_id": user["id"],
            "query_id": req.session_id,
            "duration_ms": exec_result.duration_ms,
        },
    )

    return QueryResponse(
        sql=agent_out.sql,
        explanation=agent_out.explanation,
        columns=exec_result.columns,
        rows=exec_result.rows,
        row_count=exec_result.row_count,
        truncated=exec_result.truncated,
        duration_ms=exec_result.duration_ms,
        chart=chart,
    )


@router.get("/schema", response_model=SchemaResponse)
async def get_schema(
    db: AsyncSession = Depends(get_session),
    user=Depends(get_current_user),
) -> SchemaResponse:
    tables = await load_schema(db)
    return SchemaResponse(
        schema_text=render_schema_for_llm(tables),
        table_count=len(tables),
    )


@router.post("/export")
async def export_results(
    req: ExportRequest,
    db: AsyncSession = Depends(get_session),
    user=Depends(get_current_user),
) -> Response:
    # Re-validate the SQL even though we generated it — never trust the
    # client to send back the same string we produced.
    safe = validate_sql(req.sql)
    result = await execute_sql(safe.sql, db)
    df = result.to_dataframe()

    if req.fmt == "csv":
        body = to_csv_bytes(df)
        media = "text/csv"
        filename = f"{req.filename}.csv"
    else:
        body = to_xlsx_bytes(df)
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"{req.filename}.xlsx"

    def _iter():
        yield body

    return StreamingResponse(
        _iter(),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
