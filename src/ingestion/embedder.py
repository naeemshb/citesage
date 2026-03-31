"""Embedding and Vector Store Manager.

Handles embedding chunks and storing them in ChromaDB with deduplication.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import CHROMA_DB_DIR, get_embeddings

COLLECTION_NAME = "citesage_papers"


def get_vector_store(persist_dir: str | Path | None = None) -> Chroma:
    """Get or create the ChromaDB vector store."""
    persist_dir = str(persist_dir or CHROMA_DB_DIR)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )


def get_ingested_sources(vector_store: Chroma) -> set[str]:
    """Get the set of already-ingested source filenames."""
    try:
        collection = vector_store._collection
        result = collection.get(include=["metadatas"])
        sources = set()
        for meta in result["metadatas"]:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sources
    except Exception:
        return set()


def ingest_documents(
    documents: list[Document],
    persist_dir: str | Path | None = None,
    skip_existing: bool = True,
) -> int:
    """Embed and store documents in ChromaDB.

    Args:
        documents: LangChain Documents with metadata.
        persist_dir: ChromaDB persistence directory.
        skip_existing: If True, skip documents whose source is already ingested.

    Returns:
        Number of new documents ingested.
    """
    vector_store = get_vector_store(persist_dir)

    if skip_existing:
        existing = get_ingested_sources(vector_store)
        new_docs = [d for d in documents if d.metadata.get("source") not in existing]
        skipped = len(documents) - len(new_docs)
        if skipped > 0:
            print(f"Skipping {skipped} chunks from already-ingested papers")
        documents = new_docs

    if not documents:
        print("No new documents to ingest.")
        return 0

    vector_store.add_documents(documents)
    print(f"Ingested {len(documents)} chunks into ChromaDB")
    return len(documents)


def get_ingested_papers_info(persist_dir: str | Path | None = None) -> list[dict]:
    """Get info about each ingested paper: source filename, title, chunk count."""
    vector_store = get_vector_store(persist_dir)
    try:
        collection = vector_store._collection
        result = collection.get(include=["metadatas"])
        paper_info: dict[str, dict] = {}
        for meta in result["metadatas"]:
            if not meta or "source" not in meta:
                continue
            source = meta["source"]
            if source not in paper_info:
                paper_info[source] = {
                    "source": source,
                    "title": meta.get("title", source),
                    "total_pages": meta.get("total_pages", 0),
                    "chunks": 0,
                }
            paper_info[source]["chunks"] += 1
        return list(paper_info.values())
    except Exception:
        return []


def get_paper_count(persist_dir: str | Path | None = None) -> int:
    """Get the number of unique papers in the vector store."""
    vector_store = get_vector_store(persist_dir)
    return len(get_ingested_sources(vector_store))


def get_chunk_count(persist_dir: str | Path | None = None) -> int:
    """Get the total number of chunks in the vector store."""
    vector_store = get_vector_store(persist_dir)
    try:
        return vector_store._collection.count()
    except Exception:
        return 0
