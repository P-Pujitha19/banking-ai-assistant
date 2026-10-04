"""RAG service: documents -> chunks -> embeddings -> vector store -> retriever (+ extractive helper)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from banking_ai.config.logging_config import get_logger
from banking_ai.config.settings import Settings
from banking_ai.rag.documents import Chunk, chunk_all, content_hash, load_documents
from banking_ai.rag.embeddings import analyze, build_embeddings
from banking_ai.rag.vector_store import ChromaVectorStore, NumpyVectorStore, RetrievedChunk

logger = get_logger("rag")


@dataclass
class RetrievalResult:
    """Retrieved context for a question. ``grounded`` is False when nothing relevant was found."""

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    grounded: bool = False

    def as_context(self, max_chars: int = 2500) -> str:
        """Format chunks as a numbered context block for the LLM prompt."""
        parts, used = [], 0
        for i, c in enumerate(self.chunks, 1):
            block = f"[{i}] {c.title} / {c.section}\n" + "\n".join(f"- {ln}" for ln in c.lines)
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def to_public(self) -> list[dict[str, Any]]:
        return [{"chunk_id": c.chunk_id, "title": c.title, "section": c.section, "score": c.score} for c in self.chunks]


class RagService:
    """Builds (or loads) the index and answers retrieval requests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunks: list[Chunk] = []
        self.store: Any = None
        self.backend = ""

    # ------------------------------------------------------------------ indexing
    def _choose_backend(self) -> str:
        wanted = self.settings.vector_backend
        if wanted in {"chroma", "auto"}:
            try:
                import chromadb  # noqa: F401
                import langchain_chroma  # noqa: F401
                return "chroma"
            except ImportError:
                if wanted == "chroma":
                    raise
                logger.warning("chromadb/langchain-chroma not installed; using NumPy vector store")
        return "numpy"

    def ensure_index(self, rebuild: bool = False) -> "RagService":
        """Load the persisted index if the KB is unchanged, otherwise (re)build it."""
        s = self.settings
        documents = load_documents(s.kb_dir)
        backend = self._choose_backend()
        fingerprint = content_hash(documents, s.chunk_size, s.chunk_overlap, s.embedding_provider, s.embedding_model, backend)
        manifest_path = s.vector_store_dir / "manifest.json"
        chunks_path = s.vector_store_dir / "chunks.jsonl"

        up_to_date = False
        if not rebuild and manifest_path.exists() and chunks_path.exists():
            up_to_date = json.loads(manifest_path.read_text()).get("fingerprint") == fingerprint
        if up_to_date:
            self.chunks = [Chunk(**json.loads(line)) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
        else:
            self.chunks = chunk_all(documents, s.chunk_size, s.chunk_overlap)
            s.vector_store_dir.mkdir(parents=True, exist_ok=True)
            chunks_path.write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in self.chunks), encoding="utf-8")

        embeddings = build_embeddings(s, [c.text for c in self.chunks])
        if backend == "chroma":
            self.store = ChromaVectorStore(self.chunks, embeddings, s.vector_store_dir / "chroma", rebuild=not up_to_date)
        else:
            vectors = None
            vec_path = s.vector_store_dir / "numpy" / "vectors.npy"
            if up_to_date and vec_path.exists():
                import numpy as np
                vectors = np.load(vec_path)
            self.store = NumpyVectorStore(self.chunks, embeddings, vectors)
            if vectors is None:
                self.store.save(s.vector_store_dir / "numpy")
        manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "backend": backend, "chunks": len(self.chunks),
                                             "documents": len(documents), "embedding_provider": s.embedding_provider}), encoding="utf-8")
        self.backend = backend
        logger.info("rag index ready: backend=%s chunks=%d rebuilt=%s", backend, len(self.chunks), not up_to_date)
        return self

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, query: str, k: Optional[int] = None) -> RetrievalResult:
        """Return the top-k chunks whose similarity is above ``RAG_MIN_SCORE``."""
        if self.store is None:
            self.ensure_index()
        hits = self.store.search(query, k or self.settings.rag_top_k)
        relevant = [h for h in hits if h.score >= self.settings.rag_min_score]
        return RetrievalResult(query=query, chunks=relevant, grounded=bool(relevant))


def extract_relevant_lines(query: str, result: RetrievalResult, max_lines: int = 4) -> list[tuple[str, str]]:
    """Pick the lines (from retrieved chunks) that overlap most with the query -> ``[(line, source_title)]``.

    This is the extractive answer used in demo mode (no LLM): it can only return text that exists in the KB.
    """
    q_terms = {t for t in analyze(query) if "_" not in t}
    scored: list[tuple[float, int, str, str]] = []
    order = 0
    for rank, chunk in enumerate(result.chunks):
        for line in chunk.lines:
            overlap = len(q_terms & {t for t in analyze(line) if "_" not in t})
            scored.append((overlap + chunk.score * 2 - rank * 0.05, order, line, chunk.title))
            order += 1
    best = sorted(scored, key=lambda x: -x[0])[:max_lines]
    return [(line, title) for _, _, line, title in sorted(best, key=lambda x: x[1]) if _ > 0]
