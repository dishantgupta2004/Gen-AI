"""
frontend/app.py
---------------
Streamlit chat UI for the Text-to-SQL Enterprise Analyst.

Features
--------
* Login (JWT against the FastAPI backend).
* Sidebar: backend URL, login, schema viewer, session controls.
* Chat-style transcript with:
    - The natural-language question
    - The generated SQL (collapsed by default)
    - Plain-English explanation
    - Result table (paginated)
    - Plotly chart (auto-selected by backend)
    - CSV / Excel download buttons
* Persistent session_id so the agent has conversation memory across turns.
* Light/dark theme via .streamlit/config.toml.

Run:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import json
import os
import uuid

import pandas as pd
import plotly.io as pio
import streamlit as st

from components.api_client import APIClient

# ---- Page config & styling ----

st.set_page_config(
    page_title="Text-to-SQL Analyst",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { font-family: 'IBM Plex Sans', -apple-system, sans-serif; }
      .sql-box {
          background: #0f172a; color: #e2e8f0; padding: 1rem;
          border-radius: 8px; font-family: 'JetBrains Mono', monospace;
          font-size: 0.85rem; white-space: pre-wrap; line-height: 1.5;
      }
      .explanation {
          background: #f1f5f9; padding: 0.75rem 1rem; border-radius: 6px;
          border-left: 3px solid #6366f1; margin: 0.5rem 0 1rem 0;
      }
      .meta { color: #64748b; font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Session state ----

def _init_state() -> None:
    if "client" not in st.session_state:
        st.session_state.client = APIClient()
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess-{uuid.uuid4().hex[:12]}"
    if "history" not in st.session_state:
        st.session_state.history = []  # list[dict]
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False


_init_state()
client: APIClient = st.session_state.client


# ---- Sidebar ----

with st.sidebar:
    st.title("🔎 SQL Analyst")
    st.caption("Enterprise text-to-SQL over PostgreSQL")

    # Backend URL
    backend_url = st.text_input(
        "Backend URL",
        value=os.getenv("BACKEND_URL", "http://localhost:8000"),
    )
    if backend_url != client.base:
        client.base = backend_url.rstrip("/")

    st.divider()

    # Login
    if not st.session_state.logged_in:
        st.subheader("Sign in")
        with st.form("login_form"):
            u = st.text_input("Username", value="analyst")
            p = st.text_input("Password", value="analyst", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
            if submitted:
                ok = client.login(u, p)
                if ok:
                    st.session_state.logged_in = True
                    st.success("Signed in.")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    else:
        st.success("Authenticated ✓")

        if st.button("Sign out", use_container_width=True):
            client.token = None
            st.session_state.logged_in = False
            st.rerun()

        st.divider()
        st.caption(f"Session: `{st.session_state.session_id}`")
        if st.button("New conversation", use_container_width=True):
            st.session_state.session_id = f"sess-{uuid.uuid4().hex[:12]}"
            st.session_state.history = []
            st.rerun()

        st.divider()
        with st.expander("📚 View schema"):
            try:
                schema = client.schema()
                st.caption(f"{schema['table_count']} tables")
                st.code(schema["schema_text"], language="sql")
            except Exception as e:
                st.error(f"Schema fetch failed: {e}")


# ---- Main area ----

st.title("Ask your database in plain English")

if not st.session_state.logged_in:
    st.info("Sign in on the left to start querying.")
    st.stop()

# Suggestion chips
st.caption("Try one of these or write your own:")
suggestions = [
    "Show monthly revenue growth",
    "Top 10 customers by spending",
    "Average employee performance by department",
    "Inventory items with low stock",
]
cols = st.columns(len(suggestions))
chosen = None
for i, s in enumerate(suggestions):
    if cols[i].button(s, use_container_width=True, key=f"sug-{i}"):
        chosen = s


# Render existing history
def _render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if exp := turn.get("explanation"):
            st.markdown(f"<div class='explanation'>{exp}</div>", unsafe_allow_html=True)

        with st.expander("Generated SQL", expanded=False):
            st.markdown(f"<div class='sql-box'>{turn['sql']}</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='meta'>{turn['row_count']} rows · "
            f"{turn['duration_ms']} ms"
            + ("  · result truncated" if turn["truncated"] else "")
            + "</div>",
            unsafe_allow_html=True,
        )

        df = pd.DataFrame(turn["rows"], columns=turn["columns"])
        if not df.empty:
            st.dataframe(df, use_container_width=True, height=min(420, 36 + 35 * len(df)))

        if turn.get("chart"):
            try:
                fig = pio.from_json(json.dumps(turn["chart"]))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        c1, c2, _ = st.columns([1, 1, 4])
        try:
            csv_bytes = df.to_csv(index=False).encode()
            c1.download_button(
                "⬇ CSV",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
                key=f"csv-{turn['id']}",
                use_container_width=True,
            )
        except Exception:
            pass
        try:
            xlsx = client.export(turn["sql"], fmt="xlsx", filename="results")
            c2.download_button(
                "⬇ Excel",
                data=xlsx,
                file_name="results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"xlsx-{turn['id']}",
                use_container_width=True,
            )
        except Exception:
            pass


for turn in st.session_state.history:
    _render_turn(turn)


# ---- Input ----

prompt = chosen or st.chat_input("Ask a question about your data...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking, generating SQL, executing..."):
            try:
                resp = client.query(
                    question=prompt,
                    session_id=st.session_state.session_id,
                    include_chart=True,
                )
                resp["question"] = prompt
                resp["id"] = uuid.uuid4().hex[:8]
                st.session_state.history.append(resp)
                st.rerun()
            except Exception as e:
                st.error(f"Query failed: {e}")
