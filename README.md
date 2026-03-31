# CiteSage

**Agentic research copilot with self-correcting RAG, custom MCP tool servers, and automated evaluation.**

CiteSage autonomously plans multi-step research tasks: it routes queries, retrieves from local papers or the web (arXiv + Semantic Scholar), grades document relevance, generates cited answers, and verifies its own outputs through a hallucination detection loop. It goes well beyond "chat with your PDF."

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.4+-green)
![MCP](https://img.shields.io/badge/MCP-1.2+-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Key Features

- **Agentic Workflow** — Multi-step reasoning with adaptive routing (5 routes: direct, vectorstore, web search, comparison, literature review)
- **Self-Correcting RAG** — Hallucination detection loop automatically retries with feedback when answers aren't grounded
- **Custom MCP Servers** — ArXiv and Semantic Scholar tools following the Model Context Protocol standard
- **Literature Review Generation** — Automatically generates themed, cited literature review sections
- **Paper Comparison Mode** — Structured comparison tables with per-cell citations
- **Citation Graph** — Interactive visualization of paper relationships
- **Automated Evaluation** — 30-question benchmark with groundedness, citation accuracy, routing accuracy, and completeness metrics
- **Multi-Provider LLM** — Switch between OpenAI, Anthropic, or local Ollama models
- **Streaming Agent Trace** — Watch the agent think in real-time

---

## Architecture

```
                        Streamlit Web UI
                    (chat, explorer, eval dashboard)
                              |
                    LangGraph Agent Orchestrator
                              |
            +---------+-------+--------+-----------+
            |         |       |        |           |
         Router   Retriever  Grader  Generator  Hallucination
         Node     Node       Node    Node       Checker
            |                                      |
            +--- Tool Nodes (via MCP) ---+    self-correction
            |    ArXiv MCP Server        |       loop
            |    Semantic Scholar MCP    |
            +----------------------------+
                              |
                   ChromaDB Vector Store
                              |
                  RAGAS Evaluation Engine
```

**Agent Flow:**
1. **Route** — Classify query type (concept / paper search / web discovery / comparison / literature review)
2. **Retrieve** — Search local vector store or web (arXiv + Semantic Scholar)
3. **Grade** — LLM evaluates each retrieved chunk for relevance
4. **Generate** — Produce cited answer using route-specific prompts
5. **Verify** — Hallucination checker scores groundedness; retries if below threshold
6. **Return** — Answer with citations, groundedness score, and full agent trace

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| RAG Framework | LangChain |
| Vector Store | ChromaDB |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | OpenAI / Anthropic / Ollama (configurable) |
| MCP Servers | MCP Python SDK (FastMCP) |
| PDF Parsing | PyMuPDF + pdfplumber |
| Evaluation | Custom metrics + RAGAS |
| Web UI | Streamlit |
| Deployment | Docker |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/citesage.git
cd citesage

# 2. Install dependencies
pip install -e .

# 3. Configure
cp .env.example .env
# Edit .env with your API key

# 4. Ingest papers
python scripts/ingest.py --dir data/papers/

# 5. Launch
streamlit run src/ui/app.py
```

Or with Docker:
```bash
cp .env.example .env  # edit with your key
docker compose up
# Open http://localhost:8501
```

---

## MCP Servers

CiteSage includes two custom MCP servers that can be used standalone or by the agent:

### ArXiv MCP Server
```bash
python -m src.mcp_servers.arxiv_server
# Or test: mcp dev src/mcp_servers/arxiv_server.py
```
Tools: `search_arxiv`, `get_paper_details`, `get_recent_papers`

### Semantic Scholar MCP Server
```bash
python -m src.mcp_servers.scholar_server
# Or test: mcp dev src/mcp_servers/scholar_server.py
```
Tools: `search_papers`, `get_citations`, `get_references`, `get_author_papers`

---

## Evaluation

Run the 30-question benchmark:
```bash
python scripts/evaluate.py                       # Full suite
python scripts/evaluate.py --category concept    # Single category
```

Categories: `concept`, `factual`, `discovery`, `comparison`, `literature_review`

Results are saved to `evals/experiments/` and viewable in the Evaluation Dashboard tab.

### Latest Results (24 questions, 4 papers)

| Metric | Score |
|---|---|
| **Groundedness** | 95.6% |
| **Routing Accuracy** (comparison) | 100% |
| **Routing Accuracy** (literature review) | 100% |
| **Routing Accuracy** (concept) | 83% |
| **Success Rate** | 96% (23/24) |
| **Avg Latency** | 8-26s depending on route |

---

## Project Structure

```
citesage/
├── src/
│   ├── config.py              # Multi-provider LLM config
│   ├── ingestion/             # PDF loading, chunking, embedding
│   ├── retrieval/             # Vector search with attribution
│   ├── agent/                 # LangGraph: state, nodes, graph, prompts
│   ├── mcp_servers/           # ArXiv + Semantic Scholar MCP servers
│   ├── evaluation/            # Metrics, evaluator, test dataset
│   └── ui/                    # Streamlit application
├── scripts/                   # CLI tools (ingest, evaluate)
├── tests/                     # Unit tests
├── data/papers/               # PDF storage
├── evals/experiments/         # Evaluation results
├── Dockerfile
└── docker-compose.yml
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, or `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name for the provider |
| `OPENAI_API_KEY` | — | Required for OpenAI provider + embeddings |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic provider |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |

---

## License

MIT
