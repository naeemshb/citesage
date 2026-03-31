"""ArXiv MCP Server.

Exposes tools for searching and retrieving papers from arXiv
via the Model Context Protocol.

Run standalone:
    python -m src.mcp_servers.arxiv_server
    # Or test with: mcp dev src/mcp_servers/arxiv_server.py
"""

from datetime import datetime, timedelta

import arxiv
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arxiv-research")


@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv for research papers matching the query.

    Returns title, authors, abstract, arxiv_id, published date, and PDF URL.
    """
    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

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
                "categories": paper.categories,
            })
    except Exception as e:
        if not results:
            return [{"error": f"arXiv API error: {e}"}]

    return results


@mcp.tool()
def get_paper_details(arxiv_id: str) -> dict:
    """Get full details for a specific arXiv paper by its ID (e.g., '2301.12345')."""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])

    results = list(client.results(search))
    if not results:
        return {"error": f"No paper found with ID: {arxiv_id}"}

    paper = results[0]
    return {
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "abstract": paper.summary,
        "arxiv_id": arxiv_id,
        "published": paper.published.strftime("%Y-%m-%d"),
        "updated": paper.updated.strftime("%Y-%m-%d") if paper.updated else None,
        "pdf_url": paper.pdf_url,
        "categories": paper.categories,
        "comment": paper.comment,
        "doi": paper.doi,
    }


@mcp.tool()
def get_recent_papers(
    category: str = "cs.IR",
    days: int = 30,
    max_results: int = 10,
) -> list[dict]:
    """Get recent papers from a specific arXiv category.

    Common categories: cs.IR (information retrieval), cs.LG (machine learning),
    cs.AI (artificial intelligence), cs.CR (cryptography/security),
    stat.ML (statistics/ML).
    """
    cutoff = datetime.now() - timedelta(days=days)
    query = f"cat:{category}"

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = []
    for paper in client.results(search):
        if paper.published.replace(tzinfo=None) < cutoff:
            continue
        results.append({
            "title": paper.title,
            "authors": ", ".join(a.name for a in paper.authors[:5]),
            "abstract": paper.summary[:300],
            "arxiv_id": paper.entry_id.split("/")[-1],
            "published": paper.published.strftime("%Y-%m-%d"),
            "pdf_url": paper.pdf_url,
        })

    return results


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
