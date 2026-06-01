"""
backend/agents/prompts.py
-------------------------
Prompt templates and few-shot examples.

Design notes:
  * The system prompt is *strict*: it forbids non-SELECT verbs and tells
    the model how we sanitize, so even a confused model defaults to SELECT.
  * Few-shot examples cover the most common analytical question shapes
    (aggregation, top-N, time series, joins). Each example uses a generic
    schema; at runtime we prepend the *real* schema before the examples.
  * The "explanation" prompt is separate so we can call the LLM twice:
    once for SQL, once (cheaper) for a plain-English explanation.
"""
from __future__ import annotations

SYSTEM_SQL = """You are an expert PostgreSQL analyst. Your sole job is to
translate a user's natural-language question into a single, syntactically
valid PostgreSQL SELECT statement.

Hard rules:
  1. Output ONLY raw SQL. No markdown fences, no commentary, no trailing
     semicolons, no explanation.
  2. Use ONLY the tables and columns provided in the SCHEMA section.
     Never invent identifiers. If the question cannot be answered with
     the given schema, output: SELECT 'Cannot answer with available schema' AS error
  3. The query MUST be a single SELECT (CTEs allowed via WITH).
     NEVER emit INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE,
     GRANT, REVOKE, MERGE, COPY, or SET.
  4. Always qualify columns with their table alias when joining.
  5. For aggregations, give meaningful aliases (e.g. total_revenue).
  6. For time series, order by the time column ascending.
  7. Use parameterized literals where they originate from the user
     (we will sanitize anyway, but be explicit).
  8. Default to LIMIT 1000 unless the user asks for a specific top-N
     or full export. Never return unbounded result sets.

Conversation history is provided so you can resolve follow-ups like
"only for 2024" or "show the previous result as a chart" -- in these
cases produce a NEW SELECT that incorporates the prior context.
"""


FEW_SHOT_EXAMPLES = [
    {
        "question": "Show monthly revenue for the last 12 months",
        "sql": (
            "SELECT date_trunc('month', o.created_at) AS month,\n"
            "       SUM(o.total_amount) AS total_revenue\n"
            "FROM   orders o\n"
            "WHERE  o.created_at >= NOW() - INTERVAL '12 months'\n"
            "GROUP  BY 1\n"
            "ORDER  BY 1\n"
            "LIMIT  1000"
        ),
    },
    {
        "question": "Top 10 customers by total spending",
        "sql": (
            "SELECT c.id, c.name, SUM(o.total_amount) AS total_spent\n"
            "FROM   customers c\n"
            "JOIN   orders o ON o.customer_id = c.id\n"
            "GROUP  BY c.id, c.name\n"
            "ORDER  BY total_spent DESC\n"
            "LIMIT  10"
        ),
    },
    {
        "question": "Inventory items with stock below 50",
        "sql": (
            "SELECT i.sku, i.name, i.stock_qty\n"
            "FROM   inventory i\n"
            "WHERE  i.stock_qty < 50\n"
            "ORDER  BY i.stock_qty ASC\n"
            "LIMIT  1000"
        ),
    },
    {
        "question": "Average employee performance score by department",
        "sql": (
            "SELECT d.name AS department,\n"
            "       AVG(e.performance_score)::numeric(10,2) AS avg_score,\n"
            "       COUNT(*) AS employees\n"
            "FROM   employees e\n"
            "JOIN   departments d ON d.id = e.department_id\n"
            "GROUP  BY d.name\n"
            "ORDER  BY avg_score DESC\n"
            "LIMIT  1000"
        ),
    },
]


def build_sql_prompt(
    schema_text: str,
    question: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Compose chat messages for SQL generation."""
    examples_block = "\n\n".join(
        f"Q: {ex['question']}\nSQL:\n{ex['sql']}" for ex in FEW_SHOT_EXAMPLES
    )

    user_block = (
        f"SCHEMA:\n{schema_text}\n\n"
        f"EXAMPLES:\n{examples_block}\n\n"
        f"Now answer this question. Output ONLY SQL.\n"
        f"Q: {question}"
    )

    messages = [{"role": "system", "content": SYSTEM_SQL}]
    if history:
        # Inject prior turns as plain strings so the model can resolve
        # references like "the previous query" without confusion.
        for turn in history[-4:]:  # last 2 Q/A pairs is plenty
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_block})
    return messages


SYSTEM_EXPLAIN = """You explain SQL queries to non-technical business users.
Given a SQL statement, write 2-4 short sentences in plain English describing
what the query computes, the time window if any, and the grouping. Do NOT
mention column names verbatim; speak in business terms.
"""


def build_explain_prompt(sql: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_EXPLAIN},
        {
            "role": "user",
            "content": f"User asked: {question}\n\nSQL produced:\n{sql}\n\nExplain it.",
        },
    ]
