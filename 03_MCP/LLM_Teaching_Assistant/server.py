# server.py
import os
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("JEE_NEET_Knowledge_Hub")

# --- 1. RESOURCES (Exposing Static Data) ---
@mcp.resource("syllabus://{exam_type}")
def get_syllabus(exam_type: str) -> str:
    """Provides the core topic breakdown for JEE or NEET."""
    if exam_type.lower() == "jee":
        return "Physics: Mechanics, Electrodynamics. Chemistry: Physical, Organic. Math: Calculus, Algebra."
    elif exam_type.lower() == "neet":
        return "Physics: Mechanics. Chemistry: Organic, Inorganic. Biology: Botany, Zoology, Genetics."
    return "Unknown exam type. Please choose 'jee' or 'neet'."

# --- 2. TOOLS (Exposing Executable Functions) ---
@mcp.tool()
def fetch_formulas(subject: str, chapter: str) -> str:
    """
    Fetches core formulas for a given JEE/NEET subject and chapter.
    Args:
        subject: physics, chemistry, or math
        chapter: Name of the chapter (e.g., kinematics, thermodynamics)
    """
    # In production, pull this from a local JSON file or SQLite database
    data = {
        "physics": {
            "kinematics": "1. v = u + at\n2. s = ut + 0.5*a*t^2\n3. v^2 = u^2 + 2as",
            "thermodynamics": "1. dQ = dU + dW\n2. Efficiency (eta) = 1 - T_cold/T_hot"
        }
    }
    return data.get(subject.lower(), {}).get(chapter.lower(), f"No formulas found for {chapter}.")

# --- 3. PROMPTS (Pre-configured Expert Tutors) ---
@mcp.prompt()
def socratic_tutor(student_query: str) -> str:
    """Creates a Socratic prompt template for conceptual teaching."""
    return f"""You are an elite IIT-JEE and NEET coach known for using the Socratic method. 
Never give the final numerical answer directly. Instead, ask guided questions, break down 
the physics/chemistry concepts, point out common trap options, and reference standard formulas.

Student Question: {student_query}
Provide your strategic, step-by-step guidance below:"""

if __name__ == "__main__":
    # Run the server using Standard Input/Output transport
    mcp.run(transport="stdio")