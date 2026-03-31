"""Semantic Scholar MCP Server.

Exposes tools for searching papers, citations, and references
via the Semantic Scholar API (free, no key required for basic usage).

Run standalone:
    python -m src.mcp_servers.scholar_server
    # Or test with: mcp dev src/mcp_servers/scholar_server.py
"""

import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("semantic-scholar")

BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,authors,year,citationCount,abstract,externalIds,url"

_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")


def _headers() -> dict:
    h = {"Accept": "application/json"}
    if _api_key:
        h["x-api-key"] = _api_key
    return h


@mcp.tool()
def search_papers(
    query: str,
    limit: int = 5,
    year_range: str | None = None,
) -> list[dict]:
    """Search Semantic Scholar for papers.

    Args:
        query: Search query string.
        limit: Max number of results (1-100).
        year_range: Optional year filter, e.g. '2020-2025' or '2023-'.
    """
    params = {"query": query, "limit": min(limit, 100), "fields": FIELDS}
    if year_range:
        params["year"] = year_range

    resp = httpx.get(f"{BASE_URL}/paper/search", params=params, headers=_headers(), timeout=15)
    if resp.status_code == 429:
        return [{"error": "Semantic Scholar rate limit reached. Try again in a few seconds."}]
    resp.raise_for_status()

    papers = resp.json().get("data", [])
    return [
        {
            "title": p.get("title", ""),
            "authors": ", ".join(a["name"] for a in (p.get("authors") or [])[:5]),
            "year": p.get("year"),
            "citations": p.get("citationCount", 0),
            "abstract": (p.get("abstract") or "")[:400],
            "paper_id": p.get("paperId", ""),
            "url": p.get("url", ""),
        }
        for p in papers
    ]


@mcp.tool()
def get_citations(paper_id: str, limit: int = 10) -> list[dict]:
    """Get papers that cite a given paper. Useful for finding follow-up work.

    Args:
        paper_id: Semantic Scholar paper ID.
        limit: Max number of citing papers to return.
    """
    fields = "title,authors,year,citationCount,abstract"
    params = {"fields": fields, "limit": min(limit, 100)}

    resp = httpx.get(
        f"{BASE_URL}/paper/{paper_id}/citations",
        params=params, headers=_headers(), timeout=15,
    )
    resp.raise_for_status()

    citations = resp.json().get("data", [])
    return [
        {
            "title": c["citingPaper"].get("title", ""),
            "authors": ", ".join(
                a["name"] for a in (c["citingPaper"].get("authors") or [])[:3]
            ),
            "year": c["citingPaper"].get("year"),
            "citations": c["citingPaper"].get("citationCount", 0),
            "abstract": (c["citingPaper"].get("abstract") or "")[:300],
        }
        for c in citations
        if c.get("citingPaper")
    ]


@mcp.tool()
def get_references(paper_id: str, limit: int = 10) -> list[dict]:
    """Get papers referenced by a given paper. Useful for understanding foundations.

    Args:
        paper_id: Semantic Scholar paper ID.
        limit: Max number of referenced papers to return.
    """
    fields = "title,authors,year,citationCount"
    params = {"fields": fields, "limit": min(limit, 100)}

    resp = httpx.get(
        f"{BASE_URL}/paper/{paper_id}/references",
        params=params, headers=_headers(), timeout=15,
    )
    resp.raise_for_status()

    refs = resp.json().get("data", [])
    return [
        {
            "title": r["citedPaper"].get("title", ""),
            "authors": ", ".join(
                a["name"] for a in (r["citedPaper"].get("authors") or [])[:3]
            ),
            "year": r["citedPaper"].get("year"),
            "citations": r["citedPaper"].get("citationCount", 0),
        }
        for r in refs
        if r.get("citedPaper")
    ]


@mcp.tool()
def get_author_papers(author_name: str, limit: int = 10) -> list[dict]:
    """Search for an author and return their papers.

    Args:
        author_name: Name of the author to search for.
        limit: Max number of papers to return.
    """
    # First find the author
    resp = httpx.get(
        f"{BASE_URL}/author/search",
        params={"query": author_name, "limit": 1},
        headers=_headers(), timeout=15,
    )
    resp.raise_for_status()
    authors = resp.json().get("data", [])
    if not authors:
        return [{"error": f"Author not found: {author_name}"}]

    author_id = authors[0]["authorId"]

    # Then get their papers
    fields = "title,year,citationCount,abstract"
    resp = httpx.get(
        f"{BASE_URL}/author/{author_id}/papers",
        params={"fields": fields, "limit": min(limit, 100)},
        headers=_headers(), timeout=15,
    )
    resp.raise_for_status()

    papers = resp.json().get("data", [])
    return [
        {
            "title": p.get("title", ""),
            "year": p.get("year"),
            "citations": p.get("citationCount", 0),
            "abstract": (p.get("abstract") or "")[:300],
        }
        for p in papers
    ]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
