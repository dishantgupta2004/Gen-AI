# How to run

```bash
# 1. install deps
pip install -r requirements.txt

# 2. get a free Groq key from https://console.groq.com/keys
export GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx

# 3. run the client — it will spawn the server automatically
python client/travel_client.py
```

## Optional: poke at the server directly with MCP Inspector

MCP Inspector is the official debugging UI. It connects to your server, lists
all tools / resources / prompts, and lets you invoke them manually — no LLM in
the loop. Great for catching schema bugs.

```bash
npx @modelcontextprotocol/inspector python server/travel_server.py
```

It opens a local web UI where you can:
- See the raw `initialize` response and your server's capabilities.
- Click each tool, fill in arguments, and see the JSON response.
- Fetch each resource URI and view the content.
- Invoke the `plan_vacation` prompt and inspect the resulting messages.

## What you should see (abridged)

```
══════════════════════════════════════════════════════════════════════
  1. MCP handshake (initialize)
══════════════════════════════════════════════════════════════════════
  Connected to server: travel-booking
  Server capabilities: ...tools=... resources=... prompts=...

══════════════════════════════════════════════════════════════════════
  2. Discovering TOOLS (tools/list)
══════════════════════════════════════════════════════════════════════
  • searchFlights        Search flights between two cities on a given date.
  • checkCalendar        Check the user's calendar for conflicts...
  • bookHotel            Book a hotel room. This tool has a real SIDE EFFECT.
  • searchHotels         List hotels in a city...
  • createCalendarEvent  Block off dates on the user's calendar...
  • sendEmail            Send a confirmation email to the user...

══════════════════════════════════════════════════════════════════════
  3. Discovering RESOURCES (resources/list)
══════════════════════════════════════════════════════════════════════
  • calendar://user/availability     (calendar_resource)
  • preferences://user/travel        (travel_preferences_resource)

══════════════════════════════════════════════════════════════════════
  4. Discovering PROMPTS (prompts/list)
══════════════════════════════════════════════════════════════════════
  • plan_vacation   — args: ['destination', 'departure_date', ...]

...

══════════════════════════════════════════════════════════════════════
  7. Agent loop: Groq ↔ MCP tools
══════════════════════════════════════════════════════════════════════
  -- iteration 1 --
  🔧 TOOL CALL → checkCalendar({"start_date": "2024-06-15", ...})
  ↪  result: {"range": {...}, "conflicts": [...], "free_days": [...]}

  -- iteration 2 --
  🔧 TOOL CALL → searchFlights({"origin": "NYC", ...})
  ↪  result: {...Lufthansa LH441 $612...}

  -- iteration 3 --
  🔧 TOOL CALL → searchHotels({"city": "Barcelona", "max_price_usd": 250})
  ↪  result: {...Hotel Neri $240...}

  -- iteration 4 --
  🔧 TOOL CALL → bookHotel({"hotel_name": "Hotel Neri", ...})
  ⚠️  About to call side-effect tool: bookHotel
     args: { ... }
     Approve? [y/N]: y
  ↪  result: {"booked": true, "confirmation_code": "HTL-1234567", ...}

  -- iteration 5 --
  🔧 TOOL CALL → createCalendarEvent({"title": "Barcelona trip", ...})
  ⚠️  About to call side-effect tool: createCalendarEvent
     Approve? [y/N]: y
  ↪  result: {"created": true, "event_id": "EVT-...", ...}

  -- iteration 6 --
  🔧 TOOL CALL → sendEmail({"to": "dishant@example.com", ...})
  ⚠️  About to call side-effect tool: sendEmail
     Approve? [y/N]: y
  ↪  result: {"sent": true, "message_id": "MSG-...", ...}

══════════════════════════════════════════════════════════════════════
  8. FINAL ANSWER FROM GROQ
══════════════════════════════════════════════════════════════════════
  I've planned your Barcelona trip. Here's the full itinerary:
  ✈️  Flight: Lufthansa LH441, NYC → Barcelona, June 15 ...
  🏨 Hotel: Hotel Neri (boutique), June 15–22, 7 nights at $240 ...
  📅 Calendar blocked: Barcelona trip, June 15 – June 22
  ✉️  Confirmation sent to dishant@example.com
  💰 Total spent: ...
```

## The human-in-the-loop gate in action

The client splits tools into two categories:
- **Read-only**: `searchFlights`, `searchHotels`, `checkCalendar` — run silently.
- **Side-effecting**: `bookHotel`, `createCalendarEvent`, `sendEmail` — prompt
  for y/N before running. This is exactly what the MCP spec's trust & safety
  section recommends: "Present confirmation prompts to the user for
  operations, to ensure a human is in the loop."

Denying a call (pressing anything other than `y`) sends
`{"error": "User denied this tool call."}` back to Groq so the model can
re-plan or ask the user for clarification.

Flip `REQUIRE_HUMAN_APPROVAL = False` in `travel_client.py` for automated
demos / CI.

## Things to try next

1. **Swap the prompt args** — change destination/dates/budget in
   `travel_client.py` and watch Groq re-plan.
2. **Add more tools** — e.g. `checkWeather(city, date)` returning a forecast,
   or `searchRestaurants(city, cuisine)` for dinner reservations. They'll
   appear in `tools/list` automatically thanks to the `@mcp.tool()` decorator.
3. **Add notifications** — the spec supports `notifications/tools/list_changed`
   when your tool set changes at runtime. FastMCP can emit these if you set
   `listChanged: true` in the server capabilities.
4. **Try a different transport** — swap `mcp.run(transport="stdio")` for
   `transport="streamable-http"` and deploy the server as a remote HTTP
   endpoint. Update the client to use `streamablehttp_client` instead of
   `stdio_client`.
5. **Connect this to Claude Desktop** — add an entry to your Claude Desktop
   config pointing at `server/travel_server.py` and the server's tools show
   up inside Claude Desktop itself.