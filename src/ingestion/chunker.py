"""Intelligent Text Chunker.

Splits page-level document chunks into smaller retrieval-friendly chunks,
preserving section boundaries and metadata.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.ingestion.loader import DocumentChunk
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def create_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Create a text splitter tuned for academic papers."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(
    doc_chunks: list[DocumentChunk],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split page-level chunks into smaller retrieval chunks.

    Returns LangChain Document objects ready for embedding.
    """
    splitter = create_splitter(chunk_size, chunk_overlap)

    # Convert to LangChain documents first
    lc_docs = [
        Document(page_content=dc.text, metadata=dc.metadata)
        for dc in doc_chunks
        if dc.text.strip()
    ]

    # Split
    split_docs = splitter.split_documents(lc_docs)

    # Add chunk indices per source document
    source_counters: dict[str, int] = {}
    for doc in split_docs:
        source = doc.metadata.get("source", "unknown")
        idx = source_counters.get(source, 0)
        doc.metadata["chunk_index"] = idx
        doc.metadata["chunk_id"] = f"{source}::chunk_{idx}"
        source_counters[source] = idx + 1

    print(f"Chunked {len(lc_docs)} pages into {len(split_docs)} retrieval chunks")
    return split_docs
