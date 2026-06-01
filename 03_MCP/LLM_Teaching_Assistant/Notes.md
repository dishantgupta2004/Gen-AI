# 📘 Deep-Dive Technical Notes: Model Context Protocol (MCP)

This document provides an exhaustive, step-by-step conceptual guide to the **Model Context Protocol (MCP)** as implemented in the JEE/NEET Exam Preparation Guide. It is intended for readers who want to understand not just *how* the system works, but *why* MCP exists and what architectural problem it solves.

---

## 1. What is the Model Context Protocol?

The Model Context Protocol (MCP) is an open standard that addresses a core limitation of modern Large Language Models: the **separation between generic reasoning and isolated, proprietary context**.

Historically, connecting an LLM to custom databases, local files, or specific APIs required building bespoke, brittle integration code for each platform — LangChain agents, custom wrapper code, proprietary vendor plugins. Every new model provider meant rewriting integrations from scratch.

MCP standardizes this interface. Instead of forcing developers to maintain unique API wrappers per model provider, MCP defines a clean **client–server architecture** in which any compliant server can securely expose data primitives to any compliant client.

> **Key insight:** MCP turns context provisioning into a *protocol*, not a *library*. The same MCP server can serve Claude, Groq, GPT, or any future model without modification.

---

## 2. Core Concepts: The Three Pillars

MCP organizes server capabilities into three distinct primitives: **Resources**, **Tools**, and **Prompts**.

### A. Resources — Static, Read-Only Context

Resources are data schemas exposed by the server that behave like local documents or file attachments. They represent read-only context that the LLM can reference or read directly.

- **Analogy:** A library book or a static PDF of an exam syllabus.
- **Access:** Located via unique URI patterns (e.g., `syllabus://jee`, `syllabus://neet`).

**Implementation:**

```python
@mcp.resource("syllabus://{exam_type}")
def get_syllabus(exam_type: str) -> str:
    """Expose static curriculum structure directly to the LLM."""
    ...
```

### B. Tools — Dynamic, Executable Operations

Tools represent executable functions that allow the LLM to interact with the external world or perform local computations. Unlike resources, tools accept dynamic arguments, run system actions, and return live results.

- **Analogy:** A scientific calculator or a database lookup endpoint.

**Implementation:**

```python
@mcp.tool()
def fetch_formulas(subject: str, chapter: str) -> str:
    """Dynamically extract chapter formulas from local data sources."""
    ...
```

### C. Prompts — Pre-Configured Templates and Personas

Prompts are server-side orchestration templates. Rather than embedding tutoring instructions in client code, the server manages personas and system directives centrally.

- **Analogy:** A master tutor's lesson plan or a grading rubric.

**Implementation:**

```python
@mcp.prompt()
def socratic_tutor(student_query: str) -> str:
    """Inject Socratic-method tutoring rules into the conversation."""
    ...
```

---

## 3. Step-by-Step Execution Flow

Let's trace the operation of the system chronologically, from the moment `python client.py` is invoked.

### Step 1: Server Spin-up and Standard Transport Channel

When `client.py` starts, it spawns a subprocess running `server.py` and establishes a **Stdio (Standard Input/Output) Transport Layer**:

- The client writes requests to the server's `sys.stdin`.
- The server writes responses to its own `sys.stdout`.

```python
# From client.py
server_params = StdioServerParameters(
    command="python",
    args=["server.py"]
)

async with stdio_client(server_params) as (read, write):
    ...
```

### Step 2: Protocol Handshake and Tool Discovery

Once the transport pipe is open, the client initiates a handshake via `session.initialize()` and asks the server: *"What tools do you have available?"*

The server uses Python reflection — inspecting functions decorated with `@mcp.tool()` — and returns a **JSON-Schema** representation of each tool.

```python
# From client.py
mcp_tools = await session.list_tools()
```

### Step 3: Mapping MCP Schemas to Groq Tool Definitions

Groq (like OpenAI and Anthropic) requires tool metadata in a strict function-calling format. The client maps each abstract MCP input schema into a Groq-compliant definition:

```python
groq_formatted_tools.append({
    "type": "function",
    "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.inputSchema   # MCP schema is JSON-Schema compatible
    }
})
```

Because both MCP and Groq's function-calling interface ultimately speak JSON-Schema, the conversion is near-zero overhead.

### Step 4: LLM Reasoning and Tool Invocation

The student submits a query:

> *"I'm stuck on a Physics Kinematics numerical problem. Can you help me with the formulas?"*

The client forwards this query alongside the `groq_formatted_tools` array to Groq. Groq's model recognizes that it lacks the exact curriculum formulas, halts text generation, and responds with a **tool-call request**:

```json
{
  "tool": "fetch_formulas",
  "arguments": { "subject": "physics", "chapter": "kinematics" }
}
```

### Step 5: Local Tool Execution

The client parses the tool-call request, bypasses any internet calls, and forwards the execution request directly to the local MCP server:

```python
tool_result = await session.call_tool(tool_name, arguments=tool_args)
```

The server runs `fetch_formulas("physics", "kinematics")`, retrieves the local data, and pushes the result back to the client via stdout.

### Step 6: Final Synthesis

The client appends the tool output to the chat history as a `role: tool` message and sends the updated conversation back to Groq. Groq now has the curriculum data in context and synthesizes a Socratic instruction tailored to the student.

---

## 4. Architectural Strategies for Groq Free-Tier Deployment

Running on Groq's free tier introduces strict API throttling. The architecture below mitigates these constraints.

| Challenge              | Impact on Free Tier                                                       | Remediation Strategy                                                                                          |
| ---------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Multi-Turn Latency** | Each tool call requires 2 LLM round-trips, doubling API requests.         | Use fast, low-footprint models (e.g., `llama3-8b-8192`) to preserve RPM headroom.                             |
| **Rate Limiting (429)** | Exceeding Requests-Per-Minute (RPM) limits crashes execution mid-loop.   | Implement exponential backoff and a local dictionary cache in `client.py` for repeated formula requests.      |
| **Token Exhaustion (TPM)** | Verbose prompt injections rapidly consume token allocations.            | Keep resource returns compact. Stream small, focused chunks rather than dumping entire textbook sections.     |

### Recommended Patterns

- **Caching:** Wrap `fetch_formulas` calls in an in-memory LRU cache keyed by `(subject, chapter)`.
- **Backoff:** Catch `groq.RateLimitError` and retry with exponential delay `2^n` seconds (cap at 32s).
- **Chunking:** If a resource exceeds ~500 tokens, split it across sub-tools (e.g., `fetch_formulas_basic` vs. `fetch_formulas_advanced`).

---

## 5. Summary Blueprint

By deploying MCP, the application achieves a clean separation between:

- **The cognitive engine** — Groq Cloud, handling reasoning and language synthesis.
- **The proprietary data asset** — the local Python MCP server, hosting curriculum, formulas, and pedagogy.

This separation enables future scaling without rewriting the interaction loop. The same `client.py` continues to work whether the backing store is a JSON file (today), a PostgreSQL database (tomorrow), or a vector index of millions of JEE past papers (next year).

### Why This Matters

- **Portability:** Swap Groq for Claude or GPT by changing the relay layer; the MCP server remains untouched.
- **Privacy:** Proprietary curriculum data never leaves the local environment — only narrow, relevant slices are sent as tool results.
- **Composability:** Add new tools or resources to the server without touching the client.

---

## 6. Further Reading

- [Model Context Protocol — Official Specification](https://modelcontextprotocol.io)
- [Groq Function Calling Documentation](https://console.groq.com/docs/tool-use)
- [JSON Schema Reference](https://json-schema.org)