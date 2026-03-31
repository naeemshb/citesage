"""Tests for evaluation metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    citation_accuracy,
    routing_accuracy,
    answer_has_citations,
    answer_completeness,
)


def test_citation_accuracy_perfect():
    """All citations match sources."""
    gen = "Method X works well [Source: Paper A, p.3] and Y is fast [Source: Paper B, p.5]."
    docs = [
        {"title": "Paper A", "source": "a.pdf"},
        {"title": "Paper B", "source": "b.pdf"},
    ]
    assert citation_accuracy(gen, docs) == 1.0


def test_citation_accuracy_no_citations():
    """No citations in generation should return 0."""
    gen = "Method X works well."
    docs = [{"title": "Paper A"}]
    assert citation_accuracy(gen, docs) == 0.0


def test_citation_accuracy_partial():
    """Only some citations match."""
    gen = "X [Source: Paper A, p.1] and Y [Source: Unknown Paper, p.2]."
    docs = [{"title": "Paper A"}]
    assert citation_accuracy(gen, docs) == 0.5


def test_routing_accuracy():
    assert routing_accuracy("vectorstore", "vectorstore") == 1.0
    assert routing_accuracy("web_search", "vectorstore") == 0.0


def test_answer_has_citations():
    assert answer_has_citations("result [Source: X]") is True
    assert answer_has_citations("result without citations") is False


def test_answer_completeness():
    truth = "Matrix factorization decomposes the user-item matrix into lower-rank factors."
    gen = "Matrix factorization is a technique that decomposes matrices into lower rank representations."
    score = answer_completeness(gen, truth)
    assert 0.3 < score < 1.0  # Should have significant overlap


def test_answer_completeness_empty():
    assert answer_completeness("", "something") == 0.0
    assert answer_completeness("something", "") == 1.0
