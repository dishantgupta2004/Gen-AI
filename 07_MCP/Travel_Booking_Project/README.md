# Travel Booking MCP — End-to-End Walkthrough

This project implements the "Travel Booking" example from the MCP server docs
(https://modelcontextprotocol.io/docs/learn/server-concepts) end-to-end.

## The big picture

```
┌─────────────────────┐       ┌──────────────────┐       ┌────────────────────┐
│  Groq LLM (Llama)   │◄─────►│  MCP Client /    │◄─────►│   MCP Server       │
│  (the "brain")      │ JSON  │  Host (Python)   │ JSON- │  (travel_server.py)│
│                     │ tool  │                  │ RPC   │                    │
│  Decides which      │ calls │  - Connects to   │ stdio │  Exposes:          │
│  tool to invoke     │       │    server(s)     │       │   • Tools          │
│                     │       │  - Forwards LLM  │       │   • Resources      │
│                     │       │    tool calls    │       │   • Prompts        │
│                     │       │  - Returns       │       │                    │
│                     │       │    results       │       │                    │
└─────────────────────┘       └──────────────────┘       └────────────────────┘
```

The MCP spec defines THREE building blocks a server can expose. The travel-booking
example uses all three:

| Building block | Who controls it        | Travel example                                    |
|----------------|------------------------|---------------------------------------------------|
| **Tools**      | Model-controlled       | `searchFlights`, `searchHotels`, `checkCalendar`, `bookHotel`, `createCalendarEvent`, `sendEmail` |
| **Resources**  | Application-controlled | `calendar://user/availability`, `preferences://user/travel` |
| **Prompts**    | User-controlled        | `plan_vacation(destination, departure_date, return_date, budget, travelers)` |

Plus **human-in-the-loop approval** on side-effecting tools (`bookHotel`,
`createCalendarEvent`, `sendEmail`) as recommended by the spec's trust &
safety section.

## File map

- `server/travel_server.py` — the MCP server (tools + resources + prompts)
- `client/travel_client.py` — Groq LLM + MCP client glue (the "host")
- `requirements.txt`        — dependencies
- `run.md`                  — how to run it

## Run it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_xxx...
python client/travel_client.py
```