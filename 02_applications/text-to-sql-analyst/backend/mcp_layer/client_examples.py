"""
backend/mcp_layer/client_examples.py
------------------------------------
Sample configurations for connecting MCP clients to our server.

Drop one of these into the client's config file (or copy-paste the
JSON snippets below).
"""

# ----------------------------------------------------------------------
# Claude Desktop config — file lives at:
#   macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json
#   Windows: %APPDATA%\Claude\claude_desktop_config.json
# ----------------------------------------------------------------------
CLAUDE_DESKTOP_CONFIG = r"""
{
  "mcpServers": {
    "text-to-sql-analyst": {
      "command": "python",
      "args": ["-m", "backend.mcp_layer.server"],
      "cwd": "/absolute/path/to/text-to-sql-analyst",
      "env": {
        "DATABASE_URL": "postgresql+asyncpg://readonly_user:pass@host:5432/db",
        "GROQ_API_KEY": "ignored-here",
        "JWT_SECRET": "ignored-here-but-required"
      }
    }
  }
}
"""

# ----------------------------------------------------------------------
# Cursor config — Settings -> MCP -> "Add new server"
# ----------------------------------------------------------------------
CURSOR_CONFIG = r"""
{
  "name": "text-to-sql-analyst",
  "command": "python -m backend.mcp_layer.server",
  "cwd": "/absolute/path/to/text-to-sql-analyst"
}
"""

# ----------------------------------------------------------------------
# Alternative: postgres-mcp (the official Anthropic server).
# Use this if you don't want our validator boundary and trust the LLM
# completely to issue SELECTs. We still recommend our wrapper.
# ----------------------------------------------------------------------
POSTGRES_MCP_FALLBACK = r"""
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://readonly_user:pass@host:5432/db"
      ]
    }
  }
}
"""
