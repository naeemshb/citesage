"""CLI script to ingest research papers into the vector store.

Usage:
    python scripts/ingest.py --dir data/papers/
    python scripts/ingest.py --file path/to/paper.pdf
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import load_pdf, load_directory
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import ingest_documents, get_paper_count, get_chunk_count


def main():
    parser = argparse.ArgumentParser(description="Ingest research papers into CiteSage")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=str, help="Directory containing PDF files")
    group.add_argument("--file", type=str, help="Single PDF file to ingest")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200)")
    args = parser.parse_args()

    print("=" * 60)
    print("CiteSage — Paper Ingestion")
    print("=" * 60)

    # Load
    if args.file:
        print(f"\nLoading: {args.file}")
        page_chunks = load_pdf(args.file)
    else:
        print(f"\nLoading all PDFs from: {args.dir}")
        page_chunks = load_directory(args.dir)

    if not page_chunks:
        print("No content extracted. Check your PDF files.")
        return

    # Chunk
    print(f"\nChunking with size={args.chunk_size}, overlap={args.chunk_overlap}...")
    documents = chunk_documents(page_chunks, args.chunk_size, args.chunk_overlap)

    # Embed and store
    print("\nEmbedding and storing in ChromaDB...")
    n_ingested = ingest_documents(documents)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Done! Papers in store: {get_paper_count()}, Total chunks: {get_chunk_count()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
