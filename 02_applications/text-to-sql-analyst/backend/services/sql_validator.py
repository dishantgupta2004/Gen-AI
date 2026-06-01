"""
backend/services/sql_validator.py
---------------------------------
SQL SAFETY LAYER.

This is the single most important file in the project. It blocks any
SQL that could mutate the database, escalate privileges, or exfiltrate
data through Postgres' COPY mechanism.

DEFENSE IN DEPTH (each layer would suffice alone, but we use all three):

  1. **Lexical pre-checks**: reject multiple statements, suspicious
     comment patterns, obvious injection markers.
  2. **AST validation (sqlglot)**: parse the SQL and inspect the parsed
     tree. We accept ONLY `Select` (or CTE-wrapped Select). Everything
     else (DML/DDL/DCL/utility) is rejected by type, not by string match.
  3. **Server-side enforcement** (see database/connection.py): the
     PostgreSQL session itself is `default_transaction_read_only = on`,
     so even a bypass would fail at the engine.

We also enforce a row LIMIT and an explicit allow-list of statement
types. Pattern-based blacklists alone are insufficient (attackers can
encode keywords); AST inspection is the durable layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import sqlglot
from sqlglot import exp

from backend.utils.exceptions import UnsafeQueryError
from backend.utils.logging import get_logger

logger = get_logger(__name__)

# ---- Pre-AST lexical guards -----------------------------------------------

_FORBIDDEN_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r";\s*\S",                       # multi-statement queries
        r"--",                           # line comments (often used to terminate injection)
        r"/\*.*?\*/",                    # block comments
        r"\bpg_sleep\b",                 # time-based blind injection
        r"\bcopy\b\s+.*\s+(to|from)\b",  # COPY filesystem exfiltration
        r"\bload_file\b",
        r"\bxp_cmdshell\b",
    )
)

# Statement types in a sqlglot tree we never allow at the top level.
_FORBIDDEN_NODES: Final[tuple[type[exp.Expression], ...]] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Create,
    exp.Grant,
    exp.Merge,
    exp.Command,         # generic catch-all for unparsed commands like VACUUM
    exp.Transaction,     # BEGIN/COMMIT/ROLLBACK
    exp.Set,             # SET role / SET ...
)

# Dangerous catalog tables we never want the LLM to read.
_BLOCKED_OBJECTS: Final[frozenset[str]] = frozenset({
    "pg_authid", "pg_shadow", "pg_user", "pg_roles",
    "pg_settings", "pg_hba_file_rules",
})


@dataclass
class ValidationResult:
    safe: bool
    sql: str          # possibly rewritten (e.g. LIMIT injected)
    reason: str = ""
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


def _strip(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def _lexical_check(sql: str) -> str | None:
    for pat in _FORBIDDEN_PATTERNS:
        if pat.search(sql):
            return f"Forbidden pattern: {pat.pattern}"
    return None


def _ast_check(sql: str) -> tuple[exp.Expression, str | None]:
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception as e:
        return None, f"Unparseable SQL: {e}"  # type: ignore[return-value]

    if tree is None:
        return None, "Empty parse tree"  # type: ignore[return-value]

    # Top-level must be a SELECT (or CTE wrapping a SELECT).
    if not isinstance(tree, (exp.Select, exp.Subquery, exp.With, exp.Union)):
        return tree, f"Only SELECT is allowed, got {type(tree).__name__}"

    # If it's a WITH, the body must be a Select.
    if isinstance(tree, exp.With):
        body = tree.this
        if not isinstance(body, (exp.Select, exp.Union)):
            return tree, "CTE body must be a SELECT"

    # Reject any forbidden node anywhere in the tree.
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return tree, f"Forbidden statement type: {type(node).__name__}"
        # Reject reads of blocked catalog tables.
        if isinstance(node, exp.Table):
            name = node.name.lower()
            if name in _BLOCKED_OBJECTS:
                return tree, f"Access to system catalog blocked: {name}"

    return tree, None


def _enforce_limit(tree: exp.Expression, max_rows: int) -> exp.Expression:
    """Inject a LIMIT clause if the query has none."""
    if isinstance(tree, exp.With):
        inner = tree.this
    else:
        inner = tree

    if isinstance(inner, exp.Select):
        if not inner.args.get("limit"):
            inner.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    return tree


def validate_sql(sql: str, max_rows: int = 10_000) -> ValidationResult:
    """
    Validate a candidate SQL string. Returns a ValidationResult with the
    (possibly rewritten) safe SQL or raises UnsafeQueryError on rejection.
    """
    sql = _strip(sql)
    if not sql:
        raise UnsafeQueryError("Empty SQL")

    if len(sql) > 4000:
        raise UnsafeQueryError("SQL exceeds maximum length")

    # Layer 1: lexical
    if reason := _lexical_check(sql):
        logger.warning("SQL blocked by lexical check: %s", reason)
        raise UnsafeQueryError(reason, {"sql": sql})

    # Layer 2: AST
    tree, reason = _ast_check(sql)
    if reason:
        logger.warning("SQL blocked by AST check: %s", reason)
        raise UnsafeQueryError(reason, {"sql": sql})

    # Layer 3: rewrite to enforce LIMIT
    tree = _enforce_limit(tree, max_rows)
    safe_sql = tree.sql(dialect="postgres")

    return ValidationResult(safe=True, sql=safe_sql)
