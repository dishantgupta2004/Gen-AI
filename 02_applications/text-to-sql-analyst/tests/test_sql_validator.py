"""
tests/test_sql_validator.py
---------------------------
SQL safety layer tests. This is the critical test file - it locks down
the security invariants of the application.

Run:
    pytest tests/test_sql_validator.py -v
"""
import pytest

from backend.services.sql_validator import validate_sql
from backend.utils.exceptions import UnsafeQueryError


# ---------- Allowed queries (must pass) ----------

@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "SELECT id, name FROM customers",
    "SELECT COUNT(*) FROM orders WHERE total_amount > 100",
    "WITH recent AS (SELECT * FROM orders WHERE created_at > NOW() - INTERVAL '7 days') SELECT * FROM recent",
    "SELECT c.name, SUM(o.total_amount) FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.name",
    "SELECT * FROM products UNION ALL SELECT * FROM archived_products",
])
def test_allowed_selects(sql):
    r = validate_sql(sql)
    assert r.safe is True
    # Result must still be a SELECT
    assert r.sql.upper().lstrip().startswith(("SELECT", "WITH"))


def test_limit_is_injected_when_missing():
    r = validate_sql("SELECT * FROM customers", max_rows=500)
    assert "LIMIT" in r.sql.upper()
    assert "500" in r.sql


def test_existing_limit_preserved():
    r = validate_sql("SELECT * FROM customers LIMIT 10", max_rows=500)
    # original LIMIT 10 should stay (we only inject when missing)
    assert "LIMIT 10" in r.sql.upper().replace("  ", " ")


# ---------- Blocked queries (must raise) ----------

@pytest.mark.parametrize("sql", [
    # DML
    "INSERT INTO customers (name) VALUES ('x')",
    "UPDATE customers SET name='x' WHERE id=1",
    "DELETE FROM customers WHERE id=1",
    "MERGE INTO customers USING staging ON 1=1 WHEN MATCHED THEN UPDATE SET name='x'",
    # DDL
    "DROP TABLE customers",
    "ALTER TABLE customers ADD COLUMN x int",
    "TRUNCATE TABLE customers",
    "CREATE TABLE foo (id int)",
    # DCL
    "GRANT SELECT ON customers TO public",
    # Utility / risky
    "COPY customers TO '/tmp/leak.csv'",
    "COPY customers FROM '/tmp/in.csv'",
    "SET ROLE postgres",
    "BEGIN; SELECT 1; COMMIT;",
    # Multi-statement injection
    "SELECT 1; DROP TABLE customers",
    "SELECT 1; SELECT 2",
    # Time-based blind injection
    "SELECT pg_sleep(10)",
    # Comment-based bypass attempt
    "SELECT 1 -- DROP TABLE customers",
    "SELECT 1 /* hidden */",
    # System catalog probes
    "SELECT * FROM pg_authid",
    "SELECT * FROM pg_shadow",
])
def test_blocked(sql):
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql)


def test_empty_rejected():
    with pytest.raises(UnsafeQueryError):
        validate_sql("")
    with pytest.raises(UnsafeQueryError):
        validate_sql("   ;  ")


def test_overlong_rejected():
    with pytest.raises(UnsafeQueryError):
        validate_sql("SELECT " + ("x," * 2000) + "1")


# ---------- Subtle bypass attempts ----------

def test_uppercase_does_not_bypass():
    with pytest.raises(UnsafeQueryError):
        validate_sql("DELETE FROM CUSTOMERS")


def test_mixed_case_does_not_bypass():
    with pytest.raises(UnsafeQueryError):
        validate_sql("DeLeTe FROM customers")


def test_whitespace_does_not_bypass():
    with pytest.raises(UnsafeQueryError):
        validate_sql("  DROP  TABLE  customers  ")


def test_unparseable_rejected():
    with pytest.raises(UnsafeQueryError):
        validate_sql("SELECT FROM WHERE GROUP")
