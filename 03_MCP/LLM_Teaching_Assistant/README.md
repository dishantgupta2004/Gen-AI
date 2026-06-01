# 🎓 LLM-Powered JEE/NEET Exam Preparation Guide

An AI-driven Socratic tutoring system for IIT-JEE and NEET aspirants, built on the **Model Context Protocol (MCP)** with **Groq Cloud** as the inference backend. The system decouples reasoning (cloud LLM) from curriculum data (local server), enabling deterministic, syllabus-grounded responses without sending proprietary educational content to third-party APIs.

---

## 🏛️ Architecture Overview

```text
        ┌──────────────────┐         stdio          ┌────────────────────────┐
        │                  │  ◄──────────────────   │                        │
        │   MCP Client     │                        │     MCP Server         │
        │ (Groq API Relay) │  ──────────────────►   │  (JEE/NEET Data Hub)   │
        │                  │   Tool / Resource      │                        │
        └────────┬─────────┘       Requests         └────────────────────────┘
                 │
                 │  Chat Completion &
                 │  Function Tool Schemas
                 ▼
        ┌──────────────────┐
        │    Groq Cloud    │
        │ (llama3-8b-8192) │
        └──────────────────┘
```

**Two-component system:**

1. **MCP Server (`server.py`)** — The authoritative source of curriculum knowledge. Exposes custom prompts (tutoring personas), operational tools (formula fetchers, problem retrievers), and educational resources (syllabus definitions).

2. **MCP Client (`client.py`)** — The orchestrator. Manages conversation state, opens an `stdio` communication pipe with the server, translates MCP tool schemas into Groq function-calling format, and drives the tool-execution loop against the Groq LLM.

---

## 📁 Repository Structure

```text
llm_jee_neet_guide/
│
├── .env                # Environment configuration (API keys)
├── README.md           # Project overview, installation, deployment
├── notes.md            # Conceptual architecture and MCP deep-dive
├── server.py           # MCP server: tools, prompts, resources
└── client.py           # MCP client: Groq relay and execution loop
```

---

## 🔧 Installation & Setup

### 1. Prerequisites

- Python **3.10+**
- A free-tier **Groq API key** (obtainable from [console.groq.com](https://console.groq.com))

### 2. Clone and Install Dependencies

```bash
# Create project directory
mkdir llm_jee_neet_guide
cd llm_jee_neet_guide

# Install required packages
pip install mcp groq python-dotenv
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_actual_free_tier_key_here
```

> **Note:** Never commit the `.env` file. Add it to `.gitignore` immediately.

---

## 🎮 Running the Application

To launch the interactive tutoring session:

```bash
python client.py
```

### Expected Execution Flow

1. **Client Startup** — The client spawns `server.py` as a subprocess and establishes an `stdio` communication pipe.
2. **Schema Negotiation** — The client queries the server for available tools, converts MCP schemas to Groq function-calling format, and registers them with the Groq client.
3. **Inference Loop** — When a student submits a query, the LLM identifies missing context, invokes the relevant local MCP tool, receives the curriculum data, and synthesizes a targeted pedagogical response.

### Example Interaction

```text
> I'm stuck on a kinematics numerical. Can you help me with the formulas?

[Tool Call: fetch_formulas(subject="physics", chapter="kinematics")]
[Tool Result: v = u + at, s = ut + ½at², v² = u² + 2as ...]

Tutor: Before I give you the formulas, let's think about this together.
       What quantities are given in the problem, and what are you asked
       to find? ...
```

---

## 🛡️ License

This project is licensed under the **MIT License**. See `LICENSE` for full terms.

---

## 👤 Author

Built as part of an exploration into MCP-based educational tooling.