"""
travel_client.py
================
An MCP "host + client" that uses the Groq API as the LLM brain and talks to
our travel_server.py over stdio.

MCP TERMINOLOGY REFRESHER
-------------------------
    HOST    = the AI application the user is sitting in front of (Claude
              Desktop, Cursor, our script, etc.). The host owns the LLM
              connection and the UI.
    CLIENT  = a component inside the host that maintains a 1:1 connection to
              ONE MCP server. Our host here spawns one client, for one server.
    SERVER  = the program (travel_server.py) that exposes tools/resources/
              prompts.

THE FULL FLOW this script demonstrates
--------------------------------------
    1. Host spawns the MCP server as a subprocess (stdio transport).
    2. Client + server do the MCP `initialize` handshake → negotiate
       capabilities.
    3. Host calls `tools/list`, `resources/list`, `prompts/list` to DISCOVER
       what the server offers.
    4. User picks the `plan-vacation` PROMPT → host fetches it via
       `prompts/get`.
    5. Host reads the calendar + preferences RESOURCES and prepends them as
       context.
    6. Host sends everything to Groq, advertising MCP tools as OpenAI-style
       function tools.
    7. Groq decides which TOOL to call → host forwards the call via
       `tools/call` → server runs it → result goes back to Groq.
    8. Loop until Groq stops asking for tool calls and produces a final
       answer.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# CONFIG

SERVER_SCRIPT = Path(__file__).parent.parent / "server" / "travel_server.py"

# Llama 3.3 70B on Groq supports tool calling well. 
GROQ_MODEL = "llama-3.3-70b-versatile"

# Tools with real-world side effects. Per the MCP spec's trust & safety
# guidance, these SHOULD require explicit human approval before execution.
# Read-only tools (searchFlights, searchHotels, checkCalendar) don't need it.
SIDE_EFFECT_TOOLS = {"bookHotel", "createCalendarEvent", "sendEmail"}

# Set to False to auto-approve everything (useful for scripted demos / CI).
REQUIRE_HUMAN_APPROVAL = True


def approve_tool_call(name: str, args: dict) -> bool:
    """Human-in-the-loop gate for side-effecting tools.

    The MCP spec (trust & safety section) recommends: "there SHOULD always
    be a human in the loop with the ability to deny tool invocations" and
    "Present confirmation prompts to the user for operations, to ensure a
    human is in the loop".

    For read-only tools we skip the prompt. For side-effecting tools we ask.
    """
    if not REQUIRE_HUMAN_APPROVAL or name not in SIDE_EFFECT_TOOLS:
        return True
    print(f"\n About to call side-effect tool: {name}")
    print(f"     args: {json.dumps(args, indent=2)}")
    reply = input("     Approve? [y/N]: ").strip().lower()
    return reply in ("y", "yes")


# MCP ↔ GROQ TOOL SCHEMA TRANSLATION
# Groq speaks the OpenAI function-calling format: {type:"function", function:{
# name, description, parameters(JSON Schema)}}. MCP's tools/list response
# gives us almost exactly that, just under different field names. Convert.

def mcp_tools_to_groq_tools(mcp_tools: list[Any]) -> list[dict]:
    """Translate MCP tool definitions into OpenAI/Groq function-tool format."""
    groq_tools = []
    for t in mcp_tools:
        groq_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                # MCP's inputSchema is already JSON Schema — drop it in as-is.
                "parameters": t.inputSchema,
            },
        })
    return groq_tools


# ---------------------------------------------------------------------------
# PRETTY PRINTING
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'═' * 70}\n  {title}\n{'═' * 70}")


def show_tool_call(name: str, args: dict) -> None:
    # The spec recommends "clear visual indicators when tools are invoked"
    # and "human in the loop with the ability to deny tool invocations".
    # Here we just display — in a real app you'd prompt for approval.
    print(f"\n  🔧 TOOL CALL → {name}({json.dumps(args)})")


def show_tool_result(result_text: str) -> None:
    preview = result_text if len(result_text) < 500 else result_text[:500] + "…"
    print(f"  ↪  result: {preview}")


# ---------------------------------------------------------------------------
# THE MAIN ORCHESTRATION
# ---------------------------------------------------------------------------

async def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Validate we have a Groq API key.
    # -----------------------------------------------------------------------
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: set GROQ_API_KEY env var first.", file=sys.stderr)
        sys.exit(1)
    groq = Groq(api_key=api_key)

    # -----------------------------------------------------------------------
    # 2. Spawn the MCP server as a subprocess over stdio.
    #
    # StdioServerParameters tells the SDK how to launch the server process.
    # `stdio_client(...)` returns an async context manager yielding the raw
    # read/write streams; wrapping them in ClientSession gives us the
    # high-level API (initialize, list_tools, call_tool, ...).
    # -----------------------------------------------------------------------
    server_params = StdioServerParameters(
        command=sys.executable,           # same Python we're running
        args=[str(SERVER_SCRIPT)],
    )

    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session: ClientSession = await stack.enter_async_context(
            ClientSession(read, write)
        )

        # 3. INITIALIZE handshake — capability negotiation happens here.
        #    The server advertises what it supports (tools/resources/prompts),
        #    the client advertises what IT supports (sampling, roots, etc.).
        section("1. MCP handshake (initialize)")
        init_result = await session.initialize()
        print(f"  Connected to server: {init_result.serverInfo.name}")
        print(f"  Server capabilities: {init_result.capabilities}")

        # -----------------------------------------------------------------------
        # 4. DISCOVER what the server offers.
        # -----------------------------------------------------------------------
        section("2. Discovering TOOLS (tools/list)")
        tools_resp = await session.list_tools()
        for t in tools_resp.tools:
            print(f"  • {t.name:16s}  {t.description.splitlines()[0] if t.description else ''}")

        section("3. Discovering RESOURCES (resources/list)")
        resources_resp = await session.list_resources()
        for r in resources_resp.resources:
            print(f"  • {r.uri}   ({r.name})")

        section("4. Discovering PROMPTS (prompts/list)")
        prompts_resp = await session.list_prompts()
        for p in prompts_resp.prompts:
            print(f"  • {p.name}   — args: {[a.name for a in (p.arguments or [])]}")

        # -----------------------------------------------------------------------
        # 5. The USER invokes the `plan-vacation` PROMPT.
        #    In a real UI this is a slash command. Here we do it in code.
        # -----------------------------------------------------------------------
        section("5. User invokes prompt: plan-vacation")
        prompt_args = {
            "destination": "Barcelona",
            "departure_date": "2024-06-15",
            "return_date": "2024-06-22",
            "budget": "3000",
            "travelers": "2",
        }
        print(f"  Arguments: {prompt_args}")
        prompt_resp = await session.get_prompt("plan_vacation", prompt_args)
        # get_prompt returns a list of messages; we'll use their text content.
        user_prompt_text = "\n".join(
            m.content.text for m in prompt_resp.messages if hasattr(m.content, "text")
        )
        print("  --- expanded prompt text ---")
        for line in user_prompt_text.splitlines():
            print(f"  │ {line}")

        # -----------------------------------------------------------------------
        # 6. The HOST reads RESOURCES and injects them into the system prompt.
        #    Resources are application-controlled — the MODEL doesn't fetch
        #    them; we do, because this prompt's workflow calls for it.
        # -----------------------------------------------------------------------
        section("6. Host reads resources and builds system context")
        calendar_res = await session.read_resource("calendar://user/availability")
        prefs_res = await session.read_resource("preferences://user/travel")

        calendar_text = calendar_res.contents[0].text
        prefs_text = prefs_res.contents[0].text
        print(calendar_text)
        print()
        print(prefs_text)

        system_prompt = (
            "You are a travel booking assistant with access to MCP tools.\n"
            "Use the tools to plan and book the trip. Only call one tool at a "
            "time and wait for results before deciding the next step.\n\n"
            "=== USER CALENDAR ===\n"
            f"{calendar_text}\n\n"
            "=== USER TRAVEL PREFERENCES ===\n"
            f"{prefs_text}\n"
        )

        # -----------------------------------------------------------------------
        # 7. THE AGENT LOOP — Groq plans, we execute tools, we feed results back.
        # -----------------------------------------------------------------------
        section("7. Agent loop: Groq ↔ MCP tools")

        groq_tools = mcp_tools_to_groq_tools(tools_resp.tools)

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt_text},
        ]

        MAX_ITERATIONS = 8   # safety cap — prevents infinite tool-call loops
        for step in range(1, MAX_ITERATIONS + 1):
            print(f"\n  -- iteration {step} --")

            completion = groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=groq_tools,
                tool_choice="auto",  # let the model decide whether to call a tool
            )
            msg = completion.choices[0].message

            # Record the assistant turn in history (including any tool_calls).
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])],
            })

            # If the model is done calling tools, it returns plain content.
            if not msg.tool_calls:
                section("8. FINAL ANSWER FROM GROQ")
                print(msg.content)
                return

            # Otherwise, execute every tool call the model requested.
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                show_tool_call(name, args)

                # --- HUMAN-IN-THE-LOOP GATE ---
                # Per the MCP spec, side-effecting tools should require
                # explicit user approval. Read-only tools pass through.
                if not approve_tool_call(name, args):
                    denial = {"error": "User denied this tool call."}
                    print(f"  ✋ DENIED by user")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps(denial),
                    })
                    continue

                # --- THE ACTUAL MCP tools/call REQUEST ---
                # This is where the client forwards the model's decision to
                # the server as a proper MCP JSON-RPC tools/call. The SDK
                # handles JSON-RPC framing for us.
                tool_result = await session.call_tool(name, args)

                # call_tool returns a CallToolResult with a list of content
                # blocks. For our server they're all TextContent — join them.
                result_text = "".join(
                    c.text for c in tool_result.content if hasattr(c, "text")
                )
                show_tool_result(result_text)

                # Feed the tool result back to Groq. The OpenAI/Groq spec
                # requires a role="tool" message keyed by the tool_call_id
                # the model generated.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": result_text,
                })

        print("\n  (hit max iterations without a final answer)")


if __name__ == "__main__":
    asyncio.run(main())
