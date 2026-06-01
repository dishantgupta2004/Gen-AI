-- ---------------------------------------------------------------------------
-- init-readonly.sql
--
-- Bootstraps a dedicated read-only role used by the Text-to-SQL agent.
-- Mounted into the postgres container at /docker-entrypoint-initdb.d/
-- so it runs once on first container start.
-- ---------------------------------------------------------------------------

-- The role used by the agent. NEVER reuse application roles.
CREATE ROLE analyst_ro WITH LOGIN PASSWORD 'readonly';

-- Connect + minimal schema access.
GRANT CONNECT ON DATABASE analytics TO analyst_ro;
GRANT USAGE   ON SCHEMA public       TO analyst_ro;

-- Read-only on all existing and future tables in public.
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO analyst_ro;

-- Tiny sample table so the smoke test isn't empty.
CREATE TABLE IF NOT EXISTS customers (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS orders (
    id            SERIAL PRIMARY KEY,
    customer_id   INT REFERENCES customers(id),
    total_amount  NUMERIC(12,2) NOT NULL,
    status        TEXT NOT NULL DEFAULT 'completed',
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO customers (name) VALUES
    ('Acme Corp'), ('Globex Inc'), ('Soylent Ltd');

INSERT INTO orders (customer_id, total_amount, created_at) VALUES
    (1, 1200.00, NOW() - INTERVAL '5 months'),
    (1,  800.00, NOW() - INTERVAL '3 months'),
    (2,  500.00, NOW() - INTERVAL '2 months'),
    (3, 2500.00, NOW() - INTERVAL '1 month'),
    (2,  150.00, NOW() - INTERVAL '10 days');
