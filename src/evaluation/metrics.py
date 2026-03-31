"""Custom Evaluation Metrics.

Extends standard RAG metrics with domain-specific checks:
- Citation accuracy
- Routing accuracy
- Groundedness (from agent's self-check)
"""

import re


def citation_accuracy(generation: str, source_documents: list[dict]) -> float:
    """Check if cited papers in the generation actually exist in the sources.

    Returns a score between 0.0 and 1.0.
    """
    # Extract citations from generation
    pattern = r"\[Source:\s*([^,\]]+)"
    cited_titles = [m.group(1).strip().lower() for m in re.finditer(pattern, generation)]

    if not cited_titles:
        return 0.0  # No citations = 0 accuracy (citations are expected)

    # Build set of source titles (lowercase for matching)
    source_titles = set()
    for doc in source_documents:
        title = (doc.get("title") or doc.get("source", "")).lower().strip()
        if title:
            source_titles.add(title)

    if not source_titles:
        return 0.5  # Can't verify if no sources tracked

    # Check each citation
    matched = 0
    for cited in cited_titles:
        # Fuzzy match: cited title is a substring of or contains a source title
        for source in source_titles:
            if cited in source or source in cited:
                matched += 1
                break

    return matched / len(cited_titles)


def routing_accuracy(actual_route: str, expected_route: str) -> float:
    """Check if the agent chose the correct route. Returns 1.0 or 0.0."""
    return 1.0 if actual_route == expected_route else 0.0


def answer_has_citations(generation: str) -> bool:
    """Check if the generation contains at least one citation."""
    return bool(re.search(r"\[Source:", generation))


def answer_completeness(generation: str, ground_truth: str) -> float:
    """Simple keyword overlap score between generation and ground truth.

    Not a replacement for LLM-based evaluation, but a fast heuristic.
    """
    if not generation:
        return 0.0
    if not ground_truth:
        return 1.0

    # Extract significant words (>3 chars)
    truth_words = set(
        w.lower() for w in re.findall(r"\b\w+\b", ground_truth) if len(w) > 3
    )
    gen_words = set(
        w.lower() for w in re.findall(r"\b\w+\b", generation) if len(w) > 3
    )

    if not truth_words:
        return 1.0

    overlap = truth_words & gen_words
    return len(overlap) / len(truth_words)


def compute_all_metrics(
    generation: str,
    ground_truth: str,
    source_documents: list[dict],
    actual_route: str,
    expected_route: str,
    groundedness_score: float,
) -> dict:
    """Compute all metrics for a single question-answer pair."""
    return {
        "citation_accuracy": citation_accuracy(generation, source_documents),
        "routing_accuracy": routing_accuracy(actual_route, expected_route),
        "has_citations": answer_has_citations(generation),
        "answer_completeness": answer_completeness(generation, ground_truth),
        "groundedness_score": groundedness_score,
    }
