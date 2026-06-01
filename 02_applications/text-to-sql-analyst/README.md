# Text-to-SQL Enterprise Analyst

An AI agent that connects to PostgreSQL / Supabase, translates plain-English
questions into safe SQL, executes the queries against a **read-only** database
role, and returns tables, charts, explanations, and downloads — with
multi-turn conversation memory.

```
User → Streamlit → FastAPI → LangGraph Agent (Groq · Llama 3)
       → SQL Validator (sqlglot AST)
       → PostgreSQL (RO transaction)
       → Plotly chart + CSV/XLSX export
```

The same agent stack is also exposed as a **FastMCP server**, so Claude
Desktop, Cursor, or any other MCP client can query the database under the
exact same safety rules.

---

## Highlights

- **Hard SQL safety**: lexical + sqlglot AST + read-only role + statement timeout.
  All four layers must fail before any write reaches the database. Test suite
  has 34 specifically-targeted bypass attempts; all rejected.
- **LangGraph orchestration**: explicit generate → validate → repair loop.
- **Schema introspection**: live `pg_class` / `information_schema` walk, cached
  with TTL, rendered as compact DDL for the LLM.
- **Auto-charts**: shape and dtype heuristics choose line / bar / histogram /
  grouped bar without bothering the LLM.
- **MCP layer**: FastMCP server exposes the same tools to external clients.
- **Production niceties**: JWT auth, JSON logs, request IDs, rate limiting,
  CORS, Docker multi-stage build, non-root container, healthchecks.

---

## Project layout

```
text-to-sql-analyst/
├── backend/
│   ├── main.py                    FastAPI entrypoint
│   ├── config.py                  Pydantic Settings (env-driven)
│   ├── database/
│   │   ├── connection.py          Async engine, read-only session hardening
│   │   └── schema_inspector.py    Schema introspection + LLM rendering
│   ├── agents/
│   │   ├── sql_agent.py           LangGraph generate/validate/explain
│   │   ├── prompts.py             System prompt + few-shot examples
│   │   └── memory.py              Multi-turn conversation memory
│   ├── mcp_layer/
│   │   ├── server.py              FastMCP server (tools, resources, prompts)
│   │   └── client_examples.py     Claude Desktop / Cursor config snippets
│   ├── routers/
│   │   ├── auth.py                JWT login
│   │   ├── query.py               /query, /query/schema, /query/export
│   │   └── health.py              /health
│   ├── services/
│   │   ├── sql_validator.py       SQL SAFETY LAYER (sqlglot AST)
│   │   ├── query_executor.py      Read-only async execution
│   │   ├── visualization.py       Plotly chart auto-selection
│   │   └── export.py              CSV / XLSX byte builders
│   ├── schemas/models.py          Pydantic request/response schemas
│   ├── utils/                     auth, logging, exceptions
│   └── middleware/__init__.py     RequestID + rate-limit middleware
├── frontend/
│   ├── app.py                     Streamlit chat UI
│   ├── components/api_client.py   HTTP client wrapping FastAPI
│   └── .streamlit/config.toml     Dark theme
├── tests/
│   ├── test_sql_validator.py      34 safety tests (all bypass attempts)
│   ├── test_api.py                FastAPI integration
│   └── test_visualization.py      Chart-selection unit tests
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── docker-compose.yml
│   └── init-readonly.sql          Bootstraps analyst_ro role
├── docs/
│   ├── ARCHITECTURE.md            Diagrams + sequence + security model
│   └── DEPLOYMENT.md              Render + Streamlit Cloud + Supabase
├── requirements.txt
├── requirements-frontend.txt
└── .env.example
```

---

## Quickstart (local)

### 1. Clone & set up env

```bash
cp .env.example .env
# fill in GROQ_API_KEY at minimum
# generate JWT_SECRET:
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### 2. Option A — pure Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-frontend.txt

# In one terminal:
uvicorn backend.main:app --reload

# In another:
streamlit run frontend/app.py
```

### 2. Option B — Docker

```bash
GROQ_API_KEY=gsk_... JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))") \
docker compose -f docker/docker-compose.yml up --build
```

Streamlit on http://localhost:8501, FastAPI on http://localhost:8000/docs,
PostgreSQL on `localhost:5432` (user `analyst`, password `analyst`, db `analytics`).

Demo login: `analyst` / `analyst`.

### 3. Try the chat

> "Show monthly revenue for the last 12 months"
>
> "Top 10 customers by spending"
>
> "Only show completed orders" *(follow-up — memory resolves "previous result")*
>
> "Download as Excel" *(button on the result)*

---

## Running the MCP server standalone

```bash
python -m backend.mcp_layer.server
```

Wire it into Claude Desktop via
`~/Library/Application Support/Claude/claude_desktop_config.json` — see
`backend/mcp_layer/client_examples.py` for the JSON snippet.

---

## Tests

```bash
PYTHONPATH=. pytest tests -v
```

The validator file alone covers every blocked verb (INSERT/UPDATE/DELETE/DROP/
ALTER/TRUNCATE/CREATE/GRANT/MERGE/COPY/SET/BEGIN), multi-statement injection,
comment-based bypasses, time-based blind injection (`pg_sleep`), and system
catalog probes (`pg_authid`, `pg_shadow`).

---

## Deployment

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md):

- **Backend** → Render (Docker)
- **Frontend** → Streamlit Cloud
- **Database** → Supabase (use the read-only Postgres role)

---

## Roadmap (Section 17)

- [ ] RAG over uploaded documents (semantic memory of business definitions)
- [ ] Scheduled reports (cron → email PDF)
- [ ] Slack `/ask` slash command using the same FastAPI endpoint
- [ ] Multi-agent orchestration (planner → SQL → visualization → critique)
- [ ] Voice input via Whisper
- [ ] Embedded BI dashboard (saved queries pinned to a workspace)
