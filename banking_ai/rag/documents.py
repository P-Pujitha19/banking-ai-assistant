"""Document loading and heading-aware chunking for the markdown knowledge base."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SourceDocument:
    """One markdown file of the knowledge base."""

    doc_id: str
    title: str
    text: str
    path: str


@dataclass
class Chunk:
    """A retrievable piece of a document (one section, or part of a long section)."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str  # what gets embedded: "Title - Section" header + body
    lines: list[str] = field(default_factory=list)  # body lines, used for extractive answers


def load_documents(kb_dir: Path) -> list[SourceDocument]:
    """Load every ``*.md`` file in ``kb_dir`` (the *document loader* stage of the RAG pipeline)."""
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")
    documents = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        first = next((ln for ln in text.splitlines() if ln.startswith("# ")), f"# {path.stem}")
        documents.append(SourceDocument(doc_id=path.stem, title=first[2:].strip(), text=text, path=str(path)))
    if not documents:
        raise FileNotFoundError(f"No markdown documents found in {kb_dir}")
    return documents


def chunk_document(doc: SourceDocument, chunk_size: int = 600, overlap: int = 80) -> list[Chunk]:
    """Split a document by ``##`` headings, then pack lines into chunks of ~``chunk_size`` characters.

    Heading-aware splitting keeps each chunk on one topic, which makes retrieval more precise than blind
    fixed-size windows.  ``overlap`` carries the last line(s) of a chunk into the next one when a long
    section has to be split, so a fact is never cut off from its context.
    """
    sections: list[tuple[str, list[str]]] = []
    current_title, current_lines = "Overview", []
    for raw in doc.text.splitlines():
        line = raw.strip()
        if not line or line.startswith("# "):
            continue
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title, current_lines = line[3:].strip(), []
        else:
            current_lines.append(line.lstrip("- ").strip())
    if current_lines:
        sections.append((current_title, current_lines))

    chunks: list[Chunk] = []
    for section, lines in sections:
        buffer: list[str] = []
        for line in lines:
            if buffer and sum(len(b) + 1 for b in buffer) + len(line) > chunk_size:
                chunks.append(_make_chunk(doc, section, buffer, len(chunks)))
                carry: list[str] = []
                for prev in reversed(buffer):
                    if sum(len(c) for c in carry) + len(prev) > overlap:
                        break
                    carry.insert(0, prev)
                buffer = carry
            buffer.append(line)
        if buffer:
            chunks.append(_make_chunk(doc, section, buffer, len(chunks)))
    return chunks


def _make_chunk(doc: SourceDocument, section: str, lines: list[str], index: int) -> Chunk:
    body = "\n".join(lines)
    return Chunk(chunk_id=f"{doc.doc_id}#{index}", doc_id=doc.doc_id, title=doc.title, section=section,
                 text=f"{doc.title} - {section}\n{body}", lines=list(lines))


def chunk_all(documents: list[SourceDocument], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk every document."""
    return [c for d in documents for c in chunk_document(d, chunk_size, overlap)]


def content_hash(documents: list[SourceDocument], *extra: object) -> str:
    """Fingerprint of the KB content + indexing parameters; used to decide whether to rebuild the index."""
    digest = hashlib.sha256()
    for doc in documents:
        digest.update(doc.doc_id.encode())
        digest.update(doc.text.encode())
    for item in extra:
        digest.update(repr(item).encode())
    return digest.hexdigest()[:16]
