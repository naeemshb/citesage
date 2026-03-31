"""PDF Document Loader.

Extracts text, tables, and metadata from academic PDFs.
Uses PyMuPDF for text and pdfplumber for tables.
"""

from pathlib import Path
from dataclasses import dataclass, field

import fitz  # pymupdf
import pdfplumber


@dataclass
class PaperMetadata:
    source: str
    title: str = ""
    authors: str = ""
    abstract: str = ""
    total_pages: int = 0


@dataclass
class DocumentChunk:
    text: str
    metadata: dict = field(default_factory=dict)


def extract_metadata_from_first_page(doc: fitz.Document, filename: str) -> PaperMetadata:
    """Extract title, authors, and abstract from the first page using font size heuristics."""
    meta = PaperMetadata(source=filename, total_pages=len(doc))

    if len(doc) == 0:
        return meta

    page = doc[0]
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    # Collect text spans with their font sizes
    spans = []
    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if text:
                    spans.append({"text": text, "size": span["size"], "y": span["origin"][1]})

    if not spans:
        return meta

    # Title = largest font text in the top 30% of the page, limited length
    page_height = page.rect.height
    top_spans = [s for s in spans if s["y"] < page_height * 0.3]
    if top_spans:
        max_size = max(s["size"] for s in top_spans)
        title_parts = []
        total_len = 0
        for s in top_spans:
            if abs(s["size"] - max_size) < 0.5 and total_len < 150:
                title_parts.append(s["text"])
                total_len += len(s["text"])
        meta.title = " ".join(title_parts).strip()[:200]

    # Abstract = text following "abstract" keyword
    full_text = page.get_text()
    lower_text = full_text.lower()
    abs_idx = lower_text.find("abstract")
    if abs_idx != -1:
        after_abstract = full_text[abs_idx + len("abstract"):].strip()
        # Remove leading punctuation/whitespace
        after_abstract = after_abstract.lstrip(".:- \n")
        # Take up to first double newline or 500 chars
        end = after_abstract.find("\n\n")
        if end == -1 or end > 1000:
            end = min(500, len(after_abstract))
        meta.abstract = after_abstract[:end].strip()

    return meta


def extract_tables_from_page(pdf_path: str, page_num: int) -> list[str]:
    """Extract tables from a specific page using pdfplumber, return as markdown."""
    tables_md = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return tables_md
            page = pdf.pages[page_num]
            tables = page.extract_tables()
            for table in tables:
                if not table or not table[0]:
                    continue
                # Convert to markdown table
                header = "| " + " | ".join(str(c or "") for c in table[0]) + " |"
                separator = "| " + " | ".join("---" for _ in table[0]) + " |"
                rows = []
                for row in table[1:]:
                    rows.append("| " + " | ".join(str(c or "") for c in row) + " |")
                tables_md.append("\n".join([header, separator] + rows))
    except Exception:
        pass  # Tables are supplementary; don't fail the whole load
    return tables_md


def load_pdf(pdf_path: str | Path) -> list[DocumentChunk]:
    """Load a single PDF and return a list of DocumentChunks (one per page)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    metadata = extract_metadata_from_first_page(doc, pdf_path.name)

    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Extract tables for this page
        tables = extract_tables_from_page(str(pdf_path), page_num)
        if tables:
            text += "\n\n" + "\n\n".join(tables)

        text = text.strip()
        if not text:
            continue

        chunks.append(DocumentChunk(
            text=text,
            metadata={
                "source": pdf_path.name,
                "page": page_num + 1,
                "title": metadata.title,
                "authors": metadata.authors,
                "abstract": metadata.abstract,
                "total_pages": metadata.total_pages,
            },
        ))

    doc.close()
    return chunks


def load_directory(dir_path: str | Path) -> list[DocumentChunk]:
    """Load all PDFs from a directory."""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    all_chunks = []
    pdf_files = sorted(dir_path.glob("*.pdf"))

    for pdf_file in pdf_files:
        try:
            chunks = load_pdf(pdf_file)
            all_chunks.extend(chunks)
            print(f"  Loaded {pdf_file.name}: {len(chunks)} pages")
        except Exception as e:
            print(f"  Failed to load {pdf_file.name}: {e}")

    print(f"Total: {len(all_chunks)} page-chunks from {len(pdf_files)} PDFs")
    return all_chunks
