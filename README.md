# Generative AI Workspace

A hands-on, evolving workspace covering the full GenAI stack — from LangChain fundamentals and transformer internals through production Streamlit applications and Model Context Protocol implementations.

The repository is structured as a deliberate learning path: concepts are introduced in notebooks, then applied in full applications, then extended with agentic protocols.

---

## Repository Architecture

```
Gen-AI/
├── 01_foundations/              # Notebooks & tutorials — learn the concepts
│   ├── langchain-basics/        # LangChain chains, RAG, embeddings, FAISS
│   ├── chatbot-theory/          # Conversational AI patterns & context handling
│   ├── agents_and_tools/        # ReAct agents, tool use, search pipelines
│   └── transformers/            # Attention, GPT from scratch, pre-training, fine-tuning
│
├── 02_applications/             # Deployable Streamlit apps — apply the concepts
│   ├── database-chatbot/        # Natural language → SQL  (MySQL + OpenAI/Groq)
│   ├── qa-chatbot/              # Q&A chatbot with OpenAI, Groq & Ollama backends
│   └── academic-paper-analysis/ # PDF/YouTube/URL ingestion + Claude-powered summarization
│
├── 03_MCP/                      # Model Context Protocol — protocol to production
│   ├── notifications/           # Minimal server/client reference implementation
│   ├── sampling/                # Parameter sampling demo
│   ├── LLM_Teaching_Assistant/  # Domain-specific MCP server for education
│   ├── roots/                   # Extended CLI variant with video processing
│   └── cli_project_COMPLETE/    # Full CLI chat app with tools & Anthropic API
│
└── 08_Complete_GenAI/           # Gen AI on AWS book workspace (preserved)
    ├── docs/                    # Lecture slides (Lectures 1–3)
    └── ResearchPapers/          # 14 curated papers on training, quantization & scaling
```

---

## Learning Progression

```
01_foundations/langchain-basics
    Chains, embeddings, vector stores, FAISS — the building blocks of every RAG app.
        ↓
01_foundations/chatbot-theory
    Conversational patterns, context windows, memory management.
        ↓
01_foundations/agents_and_tools
    Tool calling, ReAct loops, multi-step reasoning with search agents.
        ↓
01_foundations/transformers
    Attention mechanism, GPT architecture, pre-training & fine-tuning from scratch.
        ↓
02_applications/qa-chatbot
    First real app. Wire up a multi-provider LLM with a clean Streamlit UI.
        ↓
02_applications/database-chatbot
    Add a live database. Natural language → SQL with safety guardrails & visualizations.
        ↓
02_applications/academic-paper-analysis
    Multi-source document ingestion (PDF, YouTube, URL), advanced summarization strategies.
        ↓
03_MCP/  (notifications → sampling → LLM_Teaching_Assistant → cli_project_COMPLETE)
    Protocol layer. Extend Claude with custom tools via the Model Context Protocol.
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **LLM Frameworks** | LangChain, LangChain Community, LangChain Expression Language (LCEL) |
| **Model Providers** | Anthropic Claude (Sonnet/Haiku), OpenAI (GPT-4/3.5), Groq (Llama 3, Mixtral), Ollama (local) |
| **UI** | Streamlit, Plotly |
| **Databases** | MySQL via SQLAlchemy, FAISS vector store |
| **Document Processing** | PyPDF, YoutubeLoader, UnstructuredURLLoader, validators |
| **Protocols** | Model Context Protocol (MCP) via `mcp` SDK |
| **Package Management** | pip (foundations & apps), uv (MCP projects) |
| **Containerization** | Docker, Docker Compose |

---

## Global Setup

### Prerequisites

- Python 3.10+
- Docker & Docker Compose v2 (for containerized apps)
- API keys for the providers you plan to use

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd Gen-AI

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install shared dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

A root-level `.env.example` template is provided. Copy it to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required By | Description |
|---|---|---|
| `OPENAI_API_KEY` | database-chatbot, qa-chatbot | OpenAI API key |
| `GROQ_API_KEY` | qa-chatbot, academic-paper-analysis | Groq API key |
| `ANTHROPIC_API_KEY` | academic-paper-analysis, 03_MCP | Anthropic Claude API key |
| `LANGCHAIN_API_KEY` | qa-chatbot | LangSmith tracing (optional) |
| `MYSQL_ROOT_PASSWORD` | database-chatbot (Docker) | MySQL root password |
| `MYSQL_USER` | database-chatbot | MySQL username |
| `MYSQL_PASSWORD` | database-chatbot | MySQL password |
| `MYSQL_DATABASE` | database-chatbot | Target database name |

**For local (non-Docker) runs:** copy `.env.example` into each app directory you want to run and fill in that copy. Apps read from their own directory's `.env` via `python-dotenv`.

**For Docker:** the root `.env` is read by `docker-compose.yml` and injected as container environment variables. `MYSQL_HOST` is automatically set to the `mysql` service name inside the compose network — do not override it.

---

## Running the Applications

### Streamlit apps (local)

```bash
# Database Chatbot
cd 02_applications/database-chatbot
streamlit run app.py

# Q&A Chatbot — OpenAI + Ollama
cd 02_applications/qa-chatbot
streamlit run app.py

# Q&A Chatbot — Groq variant with search tools
cd 02_applications/qa-chatbot
streamlit run app_groq.py

# Academic Paper Analyzer
cd 02_applications/academic-paper-analysis
streamlit run app/main.py
```

### MCP projects

MCP projects use `uv` for fast, isolated environments.

```bash
# Full CLI chat app
cd 03_MCP/cli_project_COMPLETE
uv run python main.py

# Minimal examples — two terminals
cd 03_MCP/notifications
python server.py      # terminal 1
python client.py      # terminal 2
```

### Jupyter notebooks

```bash
pip install jupyter
jupyter notebook 01_foundations/
```

---

## Docker

Three applications are containerized for reproducible, dependency-isolated deployments.

### Start all services

```bash
docker compose up --build
```

| Service | URL | Description |
|---|---|---|
| `database-chatbot` | http://localhost:8501 | NL → SQL chatbot |
| `qa-chatbot` | http://localhost:8502 | Multi-backend Q&A |
| `academic-paper-analysis` | http://localhost:8503 | Paper summarizer |
| `mysql` | localhost:3306 | MySQL for database-chatbot |

### Start a single service

```bash
docker compose up database-chatbot
docker compose up qa-chatbot
docker compose up academic-paper-analysis
```

### Stop

```bash
docker compose down          # stop containers
docker compose down -v       # stop + remove MySQL data volume
```

### Environment variables for Docker

`docker-compose.yml` reads from the root `.env` file. Run `cp .env.example .env` and fill in your keys before the first `docker compose up`. See the [env vars table](#3-configure-environment-variables) for the full variable list.

---

## Module Reference

### `01_foundations/langchain-basics`

Work through the notebooks in order:

1. `01_DataIngestion.ipynb` — loading documents from files and URLs
2. `02_Text_splitting.ipynb` — chunking strategies (recursive, sentence, token)
3. `03_Embedding.ipynb` — vector representations with OpenAI embeddings
4. `04_VectorDatabase.ipynb` — FAISS indexing, similarity search & retrieval
5. `05_Langchain-openai-setup.ipynb` — end-to-end chain with an LLM
6. `SimpleLLM_LCEL/` — LangChain Expression Language patterns and serve/client setup

### `01_foundations/transformers`

Build and train a GPT model from scratch:

1. `01_tokenization.ipynb` — BPE tokenization internals
2. `02_attention.ipynb` — scaled dot-product & multi-head attention
3. `03_Implementing_GPT.ipynb` — full GPT-2-style architecture
4. `04_Pretraining.ipynb` — training loop on real text data
5. `05_finetuning.ipynb` — task-specific fine-tuning

Source modules in `src/`: `token.py`, `attention.py`, `gpt.py`, `pretraining.py`.

### `02_applications/database-chatbot`

Natural language interface to a MySQL database.

- Multi-provider LLM support: OpenAI GPT-4, Groq Llama 3
- SQL safety layer — blocks `DROP`, `DELETE`, `TRUNCATE`
- Schema explorer and sample data preview
- Plotly visualizations: histograms, scatter plots
- CSV export of query results
- Configurable entirely from the Streamlit sidebar

### `02_applications/qa-chatbot`

Two variants that demonstrate different architectural choices:

- `app.py` — OpenAI models + Ollama local models, unified via LCEL chains
- `app_groq.py` — Groq backend with DuckDuckGo, Arxiv & Wikipedia tool agents

### `02_applications/academic-paper-analysis`

Multi-source academic content summarizer powered by Claude and Groq.

- Input types: PDF upload (up to 50 MB), YouTube URL, web URL
- Summarization strategies: Stuff (short docs), Map-Reduce (long docs), Refine (iterative)
- Summary types: structured digest, bullet-point breakdown, critical analysis

### `03_MCP/`

Model Context Protocol implementations ordered from minimal to production:

| Folder | Complexity | What it demonstrates |
|---|---|---|
| `notifications/` | Minimal | Server/client handshake, notification primitives |
| `sampling/` | Basic | Parameter sampling, modern `pyproject.toml` setup |
| `LLM_Teaching_Assistant/` | Intermediate | Domain-specific MCP server for an educational use case |
| `roots/` | Intermediate | Extended CLI with added video-processing tool |
| `cli_project_COMPLETE/` | Full | Production CLI chat: tool system, Claude API, full MCP |

### `08_Complete_GenAI/`

Dedicated workspace for the **"Generative AI on AWS"** book. Do not restructure this directory.

- `docs/` — Lecture slides (Lectures 1–3)
- `ResearchPapers/` — 14 foundational papers

---

## Research Papers (`08_Complete_GenAI/ResearchPapers/`)

| # | Paper |
|---|---|
| Ch3-1 | Training Compute-Optimal Large Language Models (Chinchilla, 2022) |
| Ch3-2 | BloombergGPT: A Large Language Model for Finance (2023) |
| Ch3-3 | Llama 2: Open Foundation and Fine-Tuned Chat Models |
| Ch3-4 | Wiki-40B Multilingual Language Model Dataset (2020) |
| Ch3-5 | Exploring the Limits of Transfer Learning with T5 (2020) |
| Ch3-6 | The Pile: 800GB Diverse Dataset for Language Modeling (2020) |
| Ch3-7 | RefinedWeb Dataset for Falcon LLM (2023) |
| Ch3-8 | Scaling Laws for Neural Language Models (2020) |
| Ch3-10 | Measuring Massive Multitask Language Understanding — MMLU (2021) |
| Ch4-1 | GPTQ: Accurate Post-Training Quantization for GPT Models (2023) |
| Ch4-2 | FlashAttention: Fast, Memory-Efficient Exact Attention (2022) |
| Ch4-3 | FlashAttention-2 (2023) |
| Ch4-4 | ZeRO: Memory Optimizations Toward Trillion-Parameter Models (2020) |
| Ch4-5 | PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel (2023) |

---

## Project Status

| Module | Status |
|---|---|
| `01_foundations/langchain-basics` | Complete |
| `01_foundations/chatbot-theory` | Complete |
| `01_foundations/agents_and_tools` | Complete |
| `01_foundations/transformers` | Complete |
| `02_applications/database-chatbot` | Production-ready, Dockerized |
| `02_applications/qa-chatbot` | Production-ready, Dockerized |
| `02_applications/academic-paper-analysis` | Production-ready, Dockerized |
| `03_MCP/cli_project_COMPLETE` | Production-ready |
| `08_Complete_GenAI` | In progress (active book workspace) |
