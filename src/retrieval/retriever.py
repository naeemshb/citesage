"""Research Paper Retriever.

Wraps ChromaDB with metadata filtering, score thresholds,
and formatted source attribution. Exposed as a LangChain tool for the agent.
"""

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.ingestion.embedder import get_vector_store
from src.config import TOP_K_RESULTS


def retrieve_documents(
    query: str,
    top_k: int = TOP_K_RESULTS,
    paper_filter: str | None = None,
) -> list[dict]:
    """Search the vector store and return formatted results with attribution.

    Args:
        query: Search query string.
        top_k: Number of results to return.
        paper_filter: Optional paper source filename to filter by.

    Returns:
        List of dicts with keys: content, source, title, page, score, chunk_id
    """
    vector_store = get_vector_store()

    filter_dict = None
    if paper_filter:
        filter_dict = {"source": paper_filter}

    results: list[tuple[Document, float]] = vector_store.similarity_search_with_relevance_scores(
        query, k=top_k, filter=filter_dict
    )

    formatted = []
    for doc, score in results:
        formatted.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "title": doc.metadata.get("title", ""),
            "page": doc.metadata.get("page", 0),
            "score": round(score, 4),
            "chunk_id": doc.metadata.get("chunk_id", ""),
        })

    return formatted


@tool
def search_local_papers(query: str) -> str:
    """Search the local research paper database for relevant content.

    Use this when the question is about specific papers, methods, or results
    that may be in the ingested paper collection.

    Args:
        query: The search query describing what you're looking for.
    """
    results = retrieve_documents(query)

    if not results:
        return "No relevant documents found in the local paper database."

    output_parts = []
    for i, r in enumerate(results, 1):
        source_info = f"[Source: {r['title'] or r['source']}, page {r['page']}]"
        output_parts.append(
            f"--- Result {i} (relevance: {r['score']}) ---\n"
            f"{source_info}\n"
            f"{r['content']}\n"
        )

    return "\n".join(output_parts)
