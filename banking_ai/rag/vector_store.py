"""Vector stores: ChromaDB via LangChain (primary) and a NumPy cosine store (fallback / tests)."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from banking_ai.rag.documents import Chunk


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever with its similarity score (cosine, 0..1)."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    lines: list[str]
    score: float


class VectorStore(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedChunk]: ...


class NumpyVectorStore:
    """Brute-force cosine similarity over a normalised matrix (fast enough for thousands of chunks)."""

    backend = "numpy"

    def __init__(self, chunks: list[Chunk], embeddings: Any, vectors: np.ndarray | None = None) -> None:
        self.chunks, self.embeddings = chunks, embeddings
        matrix = vectors if vectors is not None else np.asarray(embeddings.embed_documents([c.text for c in chunks]), dtype=np.float32)
        self.matrix = self._normalise(matrix)

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.matrix)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        q = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        scores = self.matrix @ (q / norm)
        top = np.argsort(scores)[::-1][:k]
        return [_to_retrieved(self.chunks[i], float(scores[i])) for i in top]


class ChromaVectorStore:
    """ChromaDB collection managed through ``langchain_chroma.Chroma`` (persisted on disk)."""

    backend = "chroma"
    COLLECTION = "banking_kb"

    def __init__(self, chunks: list[Chunk], embeddings: Any, persist_dir: Path, rebuild: bool) -> None:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document

        self.chunks = {c.chunk_id: c for c in chunks}
        if rebuild and persist_dir.exists():
            shutil.rmtree(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.store = Chroma(collection_name=self.COLLECTION, embedding_function=embeddings,
                            persist_directory=str(persist_dir), collection_metadata={"hnsw:space": "cosine"})
        if rebuild:
            documents = [Document(page_content=c.text, metadata={"chunk_id": c.chunk_id, "doc_id": c.doc_id,
                                                                  "title": c.title, "section": c.section}) for c in chunks]
            self.store.add_documents(documents, ids=[c.chunk_id for c in chunks])

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        results = self.store.similarity_search_with_score(query, k=k)
        out = []
        for document, distance in results:
            chunk = self.chunks[document.metadata["chunk_id"]]
            out.append(_to_retrieved(chunk, max(0.0, 1.0 - float(distance))))  # cosine distance -> similarity
        return out


def _to_retrieved(chunk: Chunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk.chunk_id, chunk.doc_id, chunk.title, chunk.section, chunk.text, chunk.lines, round(score, 4))
