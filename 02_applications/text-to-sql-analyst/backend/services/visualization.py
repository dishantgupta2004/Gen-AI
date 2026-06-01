"""
backend/services/visualization.py
---------------------------------
Auto-select a chart type from result shape + dtypes, return a Plotly
figure spec as a JSON-serialisable dict.

Heuristics (deliberately simple — beats a clever model that surprises users):

  rows == 0           -> no chart (return None)
  1 row               -> KPI card (handled in UI, server returns None)
  exactly 2 cols
     temporal + num   -> line chart
     categorical+num  -> bar chart (top 50)
  exactly 3 cols
     2 categorical + num -> grouped bar
     temporal + categorical + num -> multi-line
  1 numeric col only  -> histogram
  otherwise           -> table only
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px


_TEMPORAL_DTYPES = ("datetime64", "datetime64[ns]", "datetime64[ns, UTC]")


def _is_temporal(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series) or any(
        str(series.dtype).startswith(d) for d in _TEMPORAL_DTYPES
    )


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _coerce_temporal(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Try to parse string columns that look like dates."""
    if _is_temporal(df[col]):
        return df
    try:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="raise")
    except Exception:
        pass
    return df


def build_chart(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty or len(df) < 2:
        return None

    cols = list(df.columns)

    # Try to coerce the first column to datetime if its name hints at time.
    if cols and any(k in cols[0].lower() for k in ("date", "time", "month", "year", "week")):
        df = _coerce_temporal(df, cols[0])

    numeric_cols = [c for c in cols if _is_numeric(df[c])]
    temporal_cols = [c for c in cols if _is_temporal(df[c])]
    categorical_cols = [c for c in cols if c not in numeric_cols and c not in temporal_cols]

    fig = None

    # ---- 2 columns ----
    if len(cols) == 2 and numeric_cols:
        ycol = numeric_cols[0]
        xcol = [c for c in cols if c != ycol][0]
        if xcol in temporal_cols:
            fig = px.line(df.sort_values(xcol), x=xcol, y=ycol, markers=True)
        else:
            top = df.nlargest(min(50, len(df)), ycol)
            fig = px.bar(top, x=xcol, y=ycol)

    # ---- 3 columns ----
    elif len(cols) == 3 and len(numeric_cols) == 1:
        ycol = numeric_cols[0]
        others = [c for c in cols if c != ycol]
        if any(c in temporal_cols for c in others):
            tcol = next(c for c in others if c in temporal_cols)
            ccol = next(c for c in others if c != tcol)
            fig = px.line(df.sort_values(tcol), x=tcol, y=ycol, color=ccol, markers=True)
        else:
            fig = px.bar(df, x=others[0], y=ycol, color=others[1], barmode="group")

    # ---- single numeric ----
    elif len(numeric_cols) == 1 and not categorical_cols and not temporal_cols:
        fig = px.histogram(df, x=numeric_cols[0], nbins=30)

    if fig is None:
        return None

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        title=None,
        height=420,
    )
    return fig.to_dict()
