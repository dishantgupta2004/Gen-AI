# Deployment Guide

This project ships with three independent deployment targets that map
naturally to managed PaaS:

| Component  | Target              | Why                                      |
|-----------|---------------------|------------------------------------------|
| Database  | Supabase (Postgres) | Managed Postgres + Auth + free tier.     |
| Backend   | Render (Docker)     | Native Docker, env vars, health checks.  |
| Frontend  | Streamlit Cloud     | Zero-config Streamlit hosting.           |

---

## 1. Supabase (Postgres)

1. Create a project at https://supabase.com.
2. From **Settings → Database**, grab the **connection string**.
3. In the **SQL editor**, create the dedicated read-only role:

```sql
CREATE ROLE analyst_ro LOGIN PASSWORD '<a strong password>';

GRANT CONNECT ON DATABASE postgres TO analyst_ro;
GRANT USAGE   ON SCHEMA public      TO analyst_ro;
GRANT SELECT  ON ALL TABLES IN SCHEMA public TO analyst_ro;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO analyst_ro;

-- Optional: restrict to specific schemas if you have an enterprise DB.
-- REVOKE ALL ON SCHEMA internal FROM analyst_ro;
```

4. Build the asyncpg connection string:

```
postgresql+asyncpg://analyst_ro:<password>@db.<project>.supabase.co:5432/postgres
```

5. In Supabase's **Settings → Database → Network restrictions**, whitelist
   Render's outbound IP range (Render documents these per region).

---

## 2. Render — FastAPI Backend

### 2a. Create the service

- New **Web Service** → "Deploy from a Git repository".
- **Environment**: Docker.
- **Dockerfile path**: `docker/Dockerfile.backend`.
- **Build context**: project root.
- **Start command**: (leave blank — Dockerfile's CMD handles it).
- **Health check path**: `/health`.

### 2b. Environment variables

| Key                       | Value                                                       |
|---------------------------|-------------------------------------------------------------|
| `DATABASE_URL`            | `postgresql+asyncpg://analyst_ro:...@db.<x>.supabase.co:5432/postgres` |
| `GROQ_API_KEY`            | from https://console.groq.com/keys                          |
| `GROQ_MODEL`              | `llama-3.3-70b-versatile`                                   |
| `JWT_SECRET`              | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `ENVIRONMENT`             | `production`                                                |
| `CORS_ORIGINS`            | `["https://your-frontend.streamlit.app"]`                   |
| `RATE_LIMIT_PER_MINUTE`   | `60`                                                        |
| `STATEMENT_TIMEOUT_MS`    | `15000`                                                     |
| `MAX_ROWS_RETURNED`       | `10000`                                                     |

### 2c. Scaling

- **Plan**: Start with the cheapest paid plan (Render Starter) — the free
  tier sleeps after inactivity, which makes the chat feel broken.
- **Workers**: 2 uvicorn workers per CPU is a good baseline; the Dockerfile
  defaults to 2.
- **Add Redis**: Render → New → Redis. Set `REDIS_URL` env var on the
  backend service so schema/memory/rate-limit can share state.

---

## 3. Streamlit Cloud — Frontend

1. Push the repo to GitHub.
2. https://share.streamlit.io → **New app** → point at `frontend/app.py`.
3. **Python version**: 3.12.
4. **Requirements file**: `requirements-frontend.txt`.
5. **Secrets** (Streamlit Cloud UI → Settings → Secrets):

```toml
BACKEND_URL = "https://your-backend.onrender.com"
```

The Streamlit app reads `BACKEND_URL` from env at startup; the sidebar lets
users override it temporarily if needed.

---

## 4. Verifying the deployment

```bash
# Backend up?
curl https://your-backend.onrender.com/health
# {"status":"ok","database":true}

# Auth working?
curl -X POST https://your-backend.onrender.com/api/v1/auth/login \
     -d "username=analyst&password=analyst"
# {"access_token":"...","token_type":"bearer"}

# Full query?
TOKEN=...
curl https://your-backend.onrender.com/api/v1/query \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"question":"top 5 customers by spending","session_id":"smoke-1"}'
```

---

## 5. Observability checklist for "real" production

- [ ] Replace `DEMO_USER` / `DEMO_PASSWORD` with Supabase Auth and validate the
      Supabase JWT in `utils/auth.py`.
- [ ] Wire backend logs into Datadog / Better Stack / Logtail (JSON formatter
      is on by default when `ENVIRONMENT=production`).
- [ ] Add OpenTelemetry traces around the agent run and SQL execution; the
      `duration_ms` log field is already a useful baseline.
- [ ] Audit log: persist every (user_id, question, generated_sql, result_hash,
      duration_ms) to a separate `audit.query_log` table.
- [ ] Rotate `JWT_SECRET` on a schedule. Issue tokens via a kid header so old
      and new secrets can coexist during rollout.
- [ ] Move rate limiting to Redis (`slowapi`) so it survives multi-worker
      deployments.
- [ ] Set `pool_pre_ping=True` (already on) and `pool_recycle=1800` if your
      managed Postgres terminates idle connections.
