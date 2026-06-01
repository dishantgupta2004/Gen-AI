# Architecture

## 1. High-level system

```mermaid
flowchart TD
    U[👤 User] -->|HTTPS| FE[Streamlit UI<br/>frontend/app.py]
    FE -->|JWT, REST /api/v1| BE[FastAPI Backend<br/>backend/main.py]

    subgraph Backend
      BE --> R1[query router]
      R1 --> AG[LangGraph SQL Agent<br/>backend/agents/sql_agent.py]
      AG --> SI[Schema Inspector]
      AG --> LLM[Groq · Llama 3]
      AG --> SV[SQL Validator<br/>sqlglot AST]
      R1 --> QE[Query Executor<br/>READ ONLY txn]
      R1 --> VZ[Plotly Auto-Chart]
      R1 --> EX[CSV / XLSX Export]
    end

    SI --> DB[(PostgreSQL / Supabase<br/>read-only role)]
    QE --> DB

    BE -. tool host .-> MCP[FastMCP Server<br/>backend/mcp_layer/server.py]
    MCP --> DB
    MCP <-->|stdio / SSE| EXT[External MCP clients<br/>Claude Desktop, Cursor, ...]
```

## 2. Per-request data flow

```mermaid
sequenceDiagram
    autonumber
    participant U  as User (Streamlit)
    participant A  as FastAPI /query
    participant AG as LangGraph Agent
    participant S  as Schema Cache
    participant L  as Groq Llama 3
    participant V  as SQL Validator
    participant DB as PostgreSQL (RO)

    U->>A: POST /api/v1/query { question, session_id }
    A->>AG: agent.run()
    AG->>S: load_schema() (cached, TTL 10 min)
    AG->>L: prompt = schema + few-shot + history + question
    L-->>AG: candidate SQL
    AG->>V: validate_sql(sql)
    alt unsafe
        V-->>AG: UnsafeQueryError (with reason)
        AG->>L: regenerate with self-correction (max 1)
        L-->>AG: corrected SQL
        AG->>V: validate_sql(sql)
    end
    V-->>AG: safe SQL (with LIMIT injected)
    AG->>L: build explanation
    L-->>AG: plain English
    AG-->>A: { sql, explanation }
    A->>DB: SET TRANSACTION READ ONLY; SELECT ...
    DB-->>A: rows
    A->>A: build_chart(df)
    A-->>U: { sql, explanation, rows, chart }
```

## 3. Security model (defense in depth)

```mermaid
flowchart LR
    Q[Candidate SQL] --> L1[Lexical guards<br/>multi-stmt, comments,<br/>pg_sleep, COPY]
    L1 --> L2[AST validation · sqlglot<br/>allow Select/With only<br/>reject Insert/Update/Delete/<br/>Drop/Alter/Create/Grant/etc]
    L2 --> L3[LIMIT enforcement<br/>auto-inject if missing]
    L3 --> L4[Server-side guard<br/>SET TRANSACTION READ ONLY<br/>statement_timeout]
    L4 --> L5[Database role<br/>GRANT SELECT only<br/>no GRANT INSERT/UPDATE/DELETE]
    L5 --> EX[Executed]
```

Every layer is independent. A bypass of any one is caught by the next.

## 4. Why LangGraph (not a chain)

A linear chain works until you need conditional repair:

> generate → validate → if unsafe, regenerate with the parser's feedback.

That's a graph with a conditional edge. LangGraph models it explicitly,
so the retry policy is visible in `_build_graph()` instead of buried in
`if` statements. Retries are bounded (≤ 1) and the validator's reject
reason is fed back into the next generation as context.

## 5. Why a custom FastMCP server

The off-the-shelf `@modelcontextprotocol/server-postgres` exposes raw
read-only queries. We keep our own server because:

- The same validator runs at the MCP tool boundary, so external MCP
  clients (Claude Desktop, Cursor) inherit the same safety contract.
- Schema is exposed as MCP **Resources** for lazy loading, not stuffed
  into every prompt.
- Business definitions ("what is revenue?") live as MCP **Prompts**.

The official server is provided as a fallback in
`backend/mcp_layer/client_examples.py`.

## 6. Caching & scaling notes

| Concern              | Local default              | Production swap                    |
|---------------------|----------------------------|------------------------------------|
| Schema cache         | In-process dict, 10-min TTL | Redis (so all workers share)        |
| Conversation memory  | In-process deque per session| Redis Hash per session              |
| Rate limit           | In-memory token bucket      | `slowapi` + Redis                   |
| LLM call cache       | None                        | Redis keyed on (schema_hash, q)    |
| Workers              | uvicorn `--workers 2`       | Gunicorn + uvicorn worker class    |
| Observability        | JSON logs to stdout         | OpenTelemetry → Datadog/Honeycomb  |
