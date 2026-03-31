"""Tests for MCP server tool functions (called directly, not via MCP protocol)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_arxiv_search():
    """Test that arXiv search returns structured results."""
    from src.mcp_servers.arxiv_server import search_arxiv

    results = search_arxiv("matrix factorization recommender", max_results=2)
    assert isinstance(results, list)
    assert len(results) <= 2
    if results:
        assert "title" in results[0]
        assert "arxiv_id" in results[0]
        assert "abstract" in results[0]


def test_arxiv_paper_details():
    """Test fetching a specific paper by ID."""
    from src.mcp_servers.arxiv_server import get_paper_details

    result = get_paper_details("2301.00774")  # A known paper
    assert isinstance(result, dict)
    assert "title" in result


@pytest.mark.skipif(
    True,  # Skip in CI — Semantic Scholar rate limits
    reason="Semantic Scholar API may rate-limit",
)
def test_scholar_search():
    """Test that Semantic Scholar search returns results."""
    from src.mcp_servers.scholar_server import search_papers

    results = search_papers("federated learning", limit=2)
    assert isinstance(results, list)
