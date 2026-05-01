from __future__ import annotations
from datetime import date, datetime 
from typing import Any 

from mcp.server.fastmcp import FastMCP 


### Step 1: Create the server 
### every mcp server must inherit from the FastMCP class

mcp = FastMCP(name="Travel Booking Server")

### step 2 : create a fake database of travel bookings
#### in real server, this will come from databases, external apis, google calenders etc. 

FAKE_USER_CALENDAR = [
    {"date": "2026-10-24", "event": "Diwali Preparations", "busy": True},
    {"date": "2026-10-25", "event": "Diwali Festival",     "busy": True},
    {"date": "2026-10-26", "event": None,                "busy": False},
    {"date": "2026-10-27", "event": "Bhai Dooj",          "busy": True},
]

FAKE_TRAVEL_PREFERENCES = {
    "user_email": "arjun.sharma@example.in",
    "preferred_airlines": ["IndiGo", "Air India", "Vistara"],
    "seat_preference": "window", # Popular for Himalayan/coastal routes
    "hotel_type": "heritage",    # Heritage hotels are popular in India
    "max_hotel_price_per_night_inr": 15000,
    "dietary_restrictions": ["pure vegetarian", "jain"], # Specific Indian context
    "past_favorite_cities": ["Udaipur", "Goa", "Leh"],
}


FAKE_FLIGHT_INVENTORY = {
    ("DEL", "BOM"): [ # Delhi to Mumbai
        {"flight": "6E2134", "airline": "IndiGo",    "price_inr": 5500, "stops": 0},
        {"flight": "AI865",  "airline": "Air India", "price_inr": 6200, "stops": 0},
    ],
    ("DEL", "JSA"): [ # Delhi to Jaisalmer
        {"flight": "AI2567", "airline": "Air India", "price_inr": 7800, "stops": 0},
    ],
    ("BOM", "GOI"): [ # Mumbai to Goa
        {"flight": "QP1302", "airline": "Akasa Air", "price_inr": 3200, "stops": 0},
    ],
}

FAKE_HOTELS_BY_CITY = {
    "Jaisalmer": [
        {"name": "Suryagarh",        "type": "luxury",   "price_inr": 22000, "rating": 4.9},
        {"name": "Fort Rajwada",    "type": "heritage", "price_inr": 12000, "rating": 4.5},
    ],
    "Mumbai": [
        {"name": "Taj Lands End",   "type": "luxury",   "price_inr": 25000, "rating": 4.8},
        {"name": "Ginger Hotel",    "type": "budget",   "price_inr": 4500,  "rating": 4.0},
    ],
}

### step 3 : tools (model controlled : LLM decides when to call these)

# When a client sends `tools/list`, FastMCP returns each @mcp.tool function as
# a JSON Schema entry. When the client sends `tools/call`, FastMCP validates
# arguments, runs the function, and wraps the return value in the response.

@mcp.tool()
def searchFlights(origin: str, destination: str, date: str) -> dict[str, Any]:
    """Search flights between two cities on a given date.
    
    This docstring becomes part of the tool's (tool/list) JSON Schema, which the LLM can inspect to decide when and how to call it.
    Args:
        origin:      IATA-ish city code or name for departure, e.g. "NYC".
        destination: City to fly to, e.g. "Barcelona".
        date:        Travel date in ISO format (YYYY-MM-DD).

    Returns:
        A dict with structured flight options. Returning a dict (instead of a
        plain string) lets downstream code — or another tool — consume it
        programmatically. MCP will serialize it for the LLM.
    """
    key = (origin, destination)
    options = FAKE_FLIGHT_INVENTORY.get(key, [])
    return  {
        "origin": origin,
        "destination": destination,
        "date": date,
        "results": options,
        "result_count": len(options),
    }
    
@mcp.tool()
def checkCalendar(start_date: str, end_date: str) -> dict[str, Any]:
    """Check the user's calendar for conflicts within a date range.
    Args:
        start_date: First day to check (YYYY-MM-DD), inclusive.
        end_date:   Last day to check  (YYYY-MM-DD), inclusive.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    conflicts = []
    free_days = []
    for entry in FAKE_USER_CALENDAR:
        d = date.fromisoformat(entry["date"])
        if start <= d <= end:
            if entry["busy"]:
                conflicts.append(entry)
            else:
                free_days.append(entry["date"])

    return {
        "range": {"start": start_date, "end": end_date},
        "conflicts": conflicts,
        "free_days": free_days,
        "is_range_fully_free": len(conflicts) == 0,
    }
    
    
@mcp.tool()
def bookHotel(city: str, check_in: str, check_out: str, hotel_name: str) -> dict[str, Any]:
    """Book a hotel room. 
    Args:
        city:       City where the hotel is located.
        check_in:   Check-in date (YYYY-MM-DD).
        check_out:  Check-out date (YYYY-MM-DD).
        hotel_name: Exact hotel name (from prior search results).
    """
    hotels = FAKE_HOTELS_BY_CITY.get(city, [])
    match = next((h for h in hotels if h["name"].lower() == hotel_name.lower()), None)
    if match is None:
        # Tool errors are returned as structured data, not raised exceptions,
        # so the model can read them and recover / re-plan.
        return {"booked": False, "error": f"No hotel named {hotel_name!r} in {city}."}

    nights = (date.fromisoformat(check_out) - date.fromisoformat(check_in)).days
    return {
        "booked": True,
        "confirmation_code": f"HTL-{abs(hash((hotel_name, check_in))) % 10_000_000:07d}",
        "hotel": match,
        "check_in": check_in,
        "check_out": check_out,
        "nights": nights,
        "total_usd": match["price_usd"] * nights,
    }


@mcp.tool()
def searchHotels(city: str, max_price_usd: int | None = None) -> dict[str, Any]:
    """List hotels in a city, optionally filtered by a price ceiling.

    Included so the LLM can *discover* hotel names before calling bookHotel.
    In a real server this would hit booking.com, Expedia, etc
    """
    hotels = FAKE_HOTELS_BY_CITY.get(city, [])
    if max_price_usd is not None:
        hotels = [h for h in hotels if h["price_usd"] <= max_price_usd]
    return {"city": city, "hotels": hotels, "result_count": len(hotels)}


@mcp.tool()
def createCalendarEvent(
    title: str, start_date: str, end_date: str, notes: str = ""
) -> dict[str, Any]:
    """Block off dates on the user's calendar (e.g., for the trip itself).

    This corresponds to the docs' 'Calendar Blocking' tool — after finding
    free dates via checkCalendar, the AI marks them as taken for the trip.
    """
    return {
        "created": True,
        "event_id": f"EVT-{abs(hash((title, start_date))) % 10_000_000:07d}",
        "title": title,
        "start_date": start_date,
        "end_date": end_date,
        "notes": notes,
    }
    
@mcp.tool()
def sendEmail(to: str, subject: str, body: str) -> dict[str, Any]:
    """Send a confirmation email to the user with trip details.

    This is the final tool from the docs' travel narrative:
    'sendEmail() - Sends confirmation with trip details'.

    HUMAN-IN-THE-LOOP NOTE: The MCP spec explicitly recommends that the HOST
    application present confirmation prompts for tools with side effects
    before calling them. In this project, travel_client.py enforces that
    gate before invoking sendEmail and bookHotel.
    """
    # In real life this would call SMTP, SendGrid, Resend, etc.
    return {
        "sent": True,
        "to": to,
        "subject": subject,
        "body_preview": body[:100] + ("…" if len(body) > 100 else ""),
        "message_id": f"MSG-{abs(hash((to, subject))) % 10_000_000:07d}",
    }
    
#### step 3.1 : RESOURCES (application-controlled, LLM can read but not call)

#### resources are identified by URIs. Below we
# use `calendar://` and `preferences://`

# FastMCP's @mcp.resource decorator registers each function with a URI. The
# client can then:
#   - call resources/list to see what's available
#   - call resources/read with the URI to fetch the content

@mcp.resource("calendar://user/availability")
def calendar_resource() -> str:
    """The user's calendar for the upcoming weeks.
    """
    lines = ["Upcoming calendar:"]
    for entry in FAKE_USER_CALENDAR:
        tag = "BUSY" if entry["busy"] else "FREE"
        event = entry["event"] or "-"
        lines.append(f"  {entry['date']}  [{tag}]  {event}")
    return "\n".join(lines)


@mcp.resource("preferences://user/travel")
def travel_preferences_resource() -> str:
    """The user's saved travel preferences (airlines, hotels, diet, etc.).
    """
    p = FAKE_TRAVEL_PREFERENCES
    return (
        "User travel preferences:\n"
        f"  Email:                    {p['user_email']}\n"
        f"  Preferred airlines:       {', '.join(p['preferred_airlines'])}\n"
        f"  Seat preference:          {p['seat_preference']}\n"
        f"  Hotel type:               {p['hotel_type']}\n"
        f"  Max hotel price / night:  ${p['max_hotel_price_per_night_usd']}\n"
        f"  Dietary restrictions:     {', '.join(p['dietary_restrictions'])}\n"
        f"  Past favorite cities:     {', '.join(p['past_favorite_cities'])}\n"
    )
    
    
    
### PROMPTS (User-Controlled: reusable templates)
# A prompt returns a list of messages (system / user / assistant turns) that
# the host then feeds to the LLM. They can reference resources and tools to
# orchestrate full workflows.

@mcp.prompt()
def plan_vacation(
    destination: str, 
    departure_date: str, 
    return_date: str,
    budget: str, 
    travelers: int = 1
) -> str: 
    """Plan a complete vacation: availability → flights → hotel → booking.

    This mirrors the `plan-vacation` prompt shown in the MCP docs with the
    same arguments: destination, departure_date, return_date, budget, travelers.
    The user invokes it; the prompt returns text that becomes the user-turn
    message sent to the LLM.

    The prompt deliberately tells the model to (1) read the calendar & prefs
    resources, then (2) call tools in order — showing off how prompts tie
    resources + tools together.
    """
    return (
        f"You are helping the user plan a vacation to {destination}.\n\n"
        f"Trip parameters:\n"
        f"  • Departure: {departure_date}\n"
        f"  • Return:    {return_date}\n"
        f"  • Budget:    ${budget} USD total\n"
        f"  • Travelers: {travelers}\n\n"
        f"Do this step-by-step, one tool call per turn:\n"
        f"  1. Call checkCalendar to verify the user is free on those dates.\n"
        f"  2. Call searchFlights from NYC to {destination} for {departure_date}.\n"
        f"  3. Call searchHotels in {destination} within the per-night budget.\n"
        f"  4. Call bookHotel for the best-matching option (prefer boutique\n"
        f"     within budget).\n"
        f"  5. Call createCalendarEvent to block the trip dates on the\n"
        f"     user's calendar.\n"
        f"  6. Call sendEmail to the user's email (from their preferences)\n"
        f"     with a full itinerary summary as the body.\n"
        f"  7. Produce a final human-readable itinerary summary.\n"
    )
    
# ---------------------------------------------------------------------------
# 5. RUN THE SERVER
# ---------------------------------------------------------------------------
# FastMCP supports multiple transports. `stdio` is the canonical local one:
# the server reads JSON-RPC from stdin and writes to stdout, and the client
# spawns it as a child process. Great for local dev, Claude Desktop, etc.
#
# For remote deployments you'd use `streamable-http` or SSE instead.
if __name__ == "__main__":
    mcp.run(transport="stdio")