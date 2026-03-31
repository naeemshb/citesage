"""Tests for the document ingestion pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import DocumentChunk
from src.ingestion.chunker import chunk_documents


def test_chunk_documents_basic():
    """Test that chunking produces smaller pieces with correct metadata."""
    docs = [
        DocumentChunk(
            text="A " * 600,  # ~1200 chars, should split
            metadata={"source": "test.pdf", "page": 1, "title": "Test Paper"},
        )
    ]

    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2
    assert all(c.metadata["source"] == "test.pdf" for c in chunks)
    assert all("chunk_id" in c.metadata for c in chunks)


def test_chunk_documents_preserves_metadata():
    """Test that all metadata fields are preserved through chunking."""
    docs = [
        DocumentChunk(
            text="Short text that fits in one chunk.",
            metadata={"source": "paper.pdf", "page": 3, "title": "My Paper", "authors": "Author"},
        )
    ]

    chunks = chunk_documents(docs)
    assert len(chunks) == 1
    assert chunks[0].metadata["title"] == "My Paper"
    assert chunks[0].metadata["page"] == 3
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunk_documents_empty():
    """Test that empty input produces empty output."""
    chunks = chunk_documents([])
    assert chunks == []
