import json
import re
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

from src.config import paths

SECTION_RE = re.compile(r"^\s*(Section\s+)?(\d+[A-Z]?)\b")


def load_pdf_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    """Load text page by page with page numbers."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    doc = fitz.open(pdf_path)
    pages: List[Dict[str, Any]] = []
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            pages.append({"page": i + 1, "text": page.get_text("text")})
    finally:
        doc.close()
    return pages


def structure_aware_chunk(
    pages: List[Dict[str, Any]], max_chars: int = 1200
) -> List[Dict[str, Any]]:
    """
    Simple structure-aware chunking:

    - Split text into paragraphs by double newlines.
    - Detect headings like 'Section 302 ...'.
    - Track (page, section) metadata for each chunk.
    """
    chunks: List[Dict[str, Any]] = []

    for page in pages:
        text = page["text"]
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        current = ""
        current_meta = {"page": page["page"], "section": None}

        for para in paras:
            m = SECTION_RE.match(para)
            if m:
                if current:
                    chunks.append({"text": current.strip(), "metadata": current_meta})
                sec = m.group(2)
                current = para + "\n"
                current_meta = {"page": page["page"], "section": sec}
            else:
                if len(current) + len(para) + 2 <= max_chars:
                    current += para + "\n"
                else:
                    chunks.append({"text": current.strip(), "metadata": current_meta})
                    current = para + "\n"

        if current:
            chunks.append({"text": current.strip(), "metadata": current_meta})

    return chunks


def main():
    pdf_path = Path(paths.RAW_PDF)
    out_path = Path(paths.PROCESSED_CHUNKS)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pages = load_pdf_pages(pdf_path)
    chunks = structure_aware_chunk(pages)

    data = {
        "source": pdf_path.name,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[pdf_processor] Wrote {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()
