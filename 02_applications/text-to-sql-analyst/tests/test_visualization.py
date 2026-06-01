"""
tests/test_visualization.py
"""
import pandas as pd
import pytest

from backend.services.visualization import build_chart


def test_no_chart_for_single_row():
    df = pd.DataFrame([{"x": 1}])
    assert build_chart(df) is None


def test_no_chart_for_empty():
    assert build_chart(pd.DataFrame()) is None


def test_bar_for_categorical_plus_numeric():
    df = pd.DataFrame({"category": list("abcd"), "value": [10, 20, 30, 40]})
    fig = build_chart(df)
    assert fig is not None
    assert fig["data"][0]["type"] == "bar"


def test_line_for_datetime_plus_numeric():
    df = pd.DataFrame({
        "month": pd.date_range("2024-01-01", periods=6, freq="MS"),
        "revenue": [100, 110, 130, 150, 200, 250],
    })
    fig = build_chart(df)
    assert fig is not None
    # Plotly express line chart uses scatter under the hood (mode=lines+markers).
    assert fig["data"][0]["type"] in ("scatter", "scattergl")


def test_histogram_for_single_numeric():
    df = pd.DataFrame({"score": [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]})
    fig = build_chart(df)
    assert fig is not None
    assert fig["data"][0]["type"] == "histogram"
