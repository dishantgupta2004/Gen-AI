"""
backend/services/export.py
--------------------------
CSV and XLSX exporters. Returns raw bytes so the caller can stream them
with FastAPI's StreamingResponse without buffering to disk.
"""
from __future__ import annotations

from io import BytesIO, StringIO

import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "results") -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=sheet_name[:31] or "results", index=False)
    return buf.getvalue()
