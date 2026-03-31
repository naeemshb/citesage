"""Agent Graph Nodes.

Each function is a node in the LangGraph workflow.
Nodes receive AgentState and return partial state updates.
"""

import json

from langchain_core.messages import HumanMessage, AIMessage
import httpx

from src.agent.state import AgentState
from src.agent.prompts import (
    ROUTER_PROMPT,
    GRADER_PROMPT,
    GENERATOR_PROMPT,
    COMPARE_PROMPT,
    LITERATURE_REVIEW_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
    REWRITE_PROMPT,
)
from src.config import get_llm, MAX_RETRIES
from src.retrieval.retriever import retrieve_documents


# ---------------------------------------------------------------------------
# Helper: call MCP tools directly (inline, no MCP client needed for agent use)
# ---------------------------------------------------------------------------

def _search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv directly (used by web_search node)."""
    import arxiv
    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=2)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    try:
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": ", ".join(a.name for a in paper.authors[:5]),
                "abstract": paper.summary[:500],
                "arxiv_id": paper.entry_id.split("/")[-1],
                "published": paper.published.strftime("%Y-%m-%d"),
                "pdf_url": paper.pdf_url,
            })
    except Exception:
        pass  # Return whatever we got before the error
    return results


def _search_semantic_scholar(query: str, limit: int = 5) -> list[dict]:
    """Search Semantic Scholar directly (used by web_search node)."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,authors,year,citationCount,abstract,url"
    try:
        resp = httpx.get(url, params={"query": query, "limit": limit, "fields": fields}, timeout=15)
        resp.raise_for_status()
        papers = resp.json().get("data", [])
        return [
            {
                "title": p.get("title", ""),
                "authors": ", ".join(a["name"] for a in (p.get("authors") or [])[:5]),
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "abstract": (p.get("abstract") or "")[:400],
                "url": p.get("url", ""),
            }
            for p in papers
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def route_query(state: AgentState) -> dict:
    """Classify the query to determine the best processing route."""
    llm = get_llm()
    prompt = ROUTER_PROMPT.format(question=state["question"])
    response = llm.invoke([HumanMessage(content=prompt)])

    route = response.content.strip().lower()
    valid_routes = {"vectorstore", "web_search", "direct", "compare", "literature_review"}
    if route not in valid_routes:
        route = "vectorstore"  # safe default

    return {
        "route": route,
        "trace": [f"Routed query → {route}"],
    }


def retrieve_from_vectorstore(state: AgentState) -> dict:
    """Search the local ChromaDB vector store for relevant documents."""
    results = retrieve_documents(state["question"])

    return {
        "documents": results,
        "trace": [f"Retrieved {len(results)} chunks from local papers"],
    }


def web_search(state: AgentState) -> dict:
    """Search arXiv and Semantic Scholar for papers."""
    query = state["question"]
    arxiv_results = _search_arxiv(query, max_results=5)
    scholar_results = _search_semantic_scholar(query, limit=5)

    # Convert to document format matching our schema
    documents = []
    for r in arxiv_results:
        documents.append({
            "content": f"Title: {r['title']}\nAuthors: {r['authors']}\n"
                       f"Published: {r['published']}\n\nAbstract: {r['abstract']}",
            "source": f"arXiv:{r['arxiv_id']}",
            "title": r["title"],
            "page": 0,
            "score": 1.0,
            "chunk_id": f"arxiv::{r['arxiv_id']}",
        })
    for r in scholar_results:
        documents.append({
            "content": f"Title: {r['title']}\nAuthors: {r['authors']}\n"
                       f"Year: {r['year']} | Citations: {r['citations']}\n\n"
                       f"Abstract: {r['abstract']}",
            "source": r.get("url", "Semantic Scholar"),
            "title": r["title"],
            "page": 0,
            "score": 1.0,
            "chunk_id": f"scholar::{r['title'][:50]}",
        })

    return {
        "documents": documents,
        "trace": [f"Web search: {len(arxiv_results)} arXiv + {len(scholar_results)} Scholar results"],
    }


def grade_documents(state: AgentState) -> dict:
    """Grade each retrieved document for relevance to the question."""
    llm = get_llm()
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {"documents": [], "trace": ["No documents to grade"]}

    relevant = []
    for doc in documents:
        prompt = GRADER_PROMPT.format(document=doc["content"][:500], question=question)
        response = llm.invoke([HumanMessage(content=prompt)])
        grade = response.content.strip().lower()
        if "relevant" in grade and "not_relevant" not in grade:
            relevant.append(doc)

    return {
        "documents": relevant,
        "trace": [f"Graded: {len(relevant)}/{len(documents)} documents relevant"],
    }


def generate_answer(state: AgentState) -> dict:
    """Generate a cited answer from the relevant documents."""
    llm = get_llm()
    documents = state.get("documents", [])
    question = state["question"]
    route = state.get("route", "direct")

    # Build context string — use paper titles as labels so the LLM cites them by name
    context_parts = []
    for i, doc in enumerate(documents, 1):
        paper_title = doc.get("title") or doc.get("source", "unknown")
        page_num = doc.get("page", "?")
        context_parts.append(
            f"Paper: \"{paper_title}\" (page {page_num})\n"
            f"{doc['content']}\n"
        )
    context = "\n---\n".join(context_parts) if context_parts else "No documents available."

    # Pick prompt based on route
    if route == "compare":
        prompt = COMPARE_PROMPT.format(context=context, question=question)
    elif route == "literature_review":
        prompt = LITERATURE_REVIEW_PROMPT.format(context=context, question=question)
    else:
        prompt = GENERATOR_PROMPT.format(context=context, question=question)

    # Include hallucination feedback if this is a retry
    feedback = state.get("hallucination_feedback", "")
    if feedback:
        prompt += f"\n\nPREVIOUS ATTEMPT FEEDBACK (fix these issues):\n{feedback}"

    response = llm.invoke([HumanMessage(content=prompt)])
    generation = response.content

    # Extract citations from the generated text
    import re
    citation_pattern = r"\[Source:\s*([^,\]]+)(?:,\s*p\.?\s*(\d+))?\]"
    citations = [
        {"title": m.group(1).strip(), "page": int(m.group(2)) if m.group(2) else None}
        for m in re.finditer(citation_pattern, generation)
    ]

    return {
        "generation": generation,
        "citations": citations,
        "messages": [AIMessage(content=generation)],
        "trace": [f"Generated answer with {len(citations)} citations"],
    }


def direct_answer(state: AgentState) -> dict:
    """Answer directly from LLM knowledge (no retrieval needed)."""
    llm = get_llm()
    question = state["question"]

    prompt = (
        f"You are a knowledgeable research assistant. Answer this question clearly "
        f"and concisely. Note that this answer is from general knowledge, not from "
        f"specific papers.\n\nQuestion: {question}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "generation": response.content,
        "citations": [],
        "groundedness_score": 1.0,  # N/A for direct answers
        "messages": [AIMessage(content=response.content)],
        "trace": ["Answered directly from model knowledge"],
    }


def check_hallucination(state: AgentState) -> dict:
    """Check if the generated answer is grounded in the source documents."""
    llm = get_llm()
    documents = state.get("documents", [])
    generation = state.get("generation", "")

    if not documents:
        return {
            "groundedness_score": 1.0,
            "hallucination_feedback": "",
            "trace": ["Skipped hallucination check (no source documents)"],
        }

    # Summarize documents for the check
    doc_summaries = []
    for doc in documents[:10]:
        source = doc.get("title") or doc.get("source", "unknown")
        doc_summaries.append(f"[{source}]: {doc['content'][:300]}")
    doc_text = "\n\n".join(doc_summaries)

    prompt = HALLUCINATION_CHECK_PROMPT.format(documents=doc_text, generation=generation)
    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse JSON response
    try:
        # Extract JSON from response (handle markdown code blocks)
        text = response.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        result = {"is_grounded": True, "groundedness_score": 0.7, "feedback": "", "unsupported_claims": []}

    score = float(result.get("groundedness_score", 0.7))
    feedback = result.get("feedback", "")

    return {
        "groundedness_score": score,
        "hallucination_feedback": feedback if score < 0.7 else "",
        "trace": [f"Groundedness check: {score:.2f}"],
    }


def rewrite_query(state: AgentState) -> dict:
    """Rewrite the query when retrieval returned insufficient results."""
    llm = get_llm()
    question = state["question"]
    feedback = state.get("hallucination_feedback", "No relevant results found.")

    prompt = REWRITE_PROMPT.format(question=question, feedback=feedback)
    response = llm.invoke([HumanMessage(content=prompt)])
    new_question = response.content.strip()

    retry = state.get("retry_count", 0) + 1

    return {
        "question": new_question,
        "retry_count": retry,
        "trace": [f"Rewrote query (attempt {retry}): {new_question[:80]}..."],
    }


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def should_route_to(state: AgentState) -> str:
    """Decide which node to execute based on the route."""
    route = state.get("route", "vectorstore")
    if route == "direct":
        return "direct_answer"
    elif route == "web_search":
        return "web_search"
    else:
        # vectorstore, compare, literature_review all start with retrieval
        return "retrieve"


def should_retry_or_finish(state: AgentState) -> str:
    """After hallucination check, decide whether to retry or return the answer."""
    score = state.get("groundedness_score", 1.0)
    retries = state.get("retry_count", 0)

    if score >= 0.7 or retries >= MAX_RETRIES:
        return "finish"
    else:
        return "retry"


def has_relevant_docs(state: AgentState) -> str:
    """After grading, check if we have any relevant documents."""
    docs = state.get("documents", [])
    retries = state.get("retry_count", 0)

    if docs:
        return "generate"
    elif retries < MAX_RETRIES:
        return "rewrite"
    else:
        return "generate"  # generate with whatever we have
